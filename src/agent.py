import asyncio
import json
import logging
from typing import Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess, TurnHandlingOptions, cli, inference, metrics
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from backend import BackendClient, UserState
from stages import stage_to_agent

load_dotenv(".env.local")

logger = logging.getLogger("loan-agent")

OBSERVER_PREFIX = "observer_"
PARTICIPANT_WAIT_TIMEOUT = 120.0

COLD_CALL_MAX_SECONDS = 180
_MAX_DURATION_STAGES = {"cold_call"}


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="loan-lifecycle-agent")
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")

    # --- 1. Read metadata ---
    metadata = json.loads(ctx.room.metadata or "{}")
    user_id = metadata.get("user_id")
    stage_override = metadata.get("current_stage")

    if not user_id:
        logger.error("No user_id in room metadata.")
        return

    # --- 2. Build user state directly from room metadata (no DB fetch) ---
    backend = BackendClient()
    user_state = UserState(
        user_id=user_id,
        current_stage=stage_override or metadata.get("current_stage", "cold_call"),
        name=metadata.get("name", ""),
        phone=metadata.get("phone", ""),
        ref_source=metadata.get("ref_source", "other"),
        loan_amount_interest=metadata.get("loan_amount_interest", 150000),
        city=metadata.get("city", ""),
        cold_call_attempts=metadata.get("cold_call_attempts", 0),
    )

    active_stage = user_state.current_stage
    logger.info(f"Stage '{active_stage}' | user={user_id}")

    # --- 3. Resolve agent ---
    agent = stage_to_agent(active_stage, user_state, backend)
    if agent is None:
        logger.error(f"No agent for stage '{active_stage}' — backend-only stage?")
        return

    # --- 4. Wait for participant ---
    active_participant = await _wait_for_participant(ctx)
    if not active_participant:
        logger.warning("No participant joined.")
        return

    # --- 5. Build session ---
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general", language="multi"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="elevenlabs/eleven_flash_v2_5",
            voice="EXAVITQu4vr4xnSDxMaL",
            language="hi",
        ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()
    session.on("metrics_collected")(
        lambda ev: (metrics.log_metrics(ev.metrics), usage_collector.collect(ev.metrics))
    )
    session.on("close")(
        lambda ev: logger.info(f"Session closed. Error: {ev.error or 'None'}")
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            participant_identity=active_participant.identity,
            audio_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
            text_enabled=True,
            close_on_disconnect=False,
        ),
    )

    # --- 6. Max duration enforcement for timed stages ---
    timeout_task: Optional[asyncio.Task] = None
    if active_stage in _MAX_DURATION_STAGES:
        timeout_task = asyncio.create_task(
            _enforce_max_duration(session, COLD_CALL_MAX_SECONDS)
        )

    async def _on_shutdown() -> None:
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(_on_shutdown)
    logger.info(f"Session live | user={user_state.name} | stage={active_stage}")


async def _enforce_max_duration(session: AgentSession, seconds: int) -> None:
    """
    Hard cap on call duration. After `seconds`, say a polite closing and let the
    call drop naturally — we don't force-close the room from the agent side.
    """
    await asyncio.sleep(seconds)
    logger.info(f"Max call duration ({seconds}s) reached — closing gracefully")
    try:
        await session.say(
            "Theek hai — main baad mein call karti hoon. Dhanyawaad!",
            allow_interruptions=False,
        )
    except Exception:
        pass  # session may already be closing


async def _wait_for_participant(ctx: JobContext) -> Optional[rtc.RemoteParticipant]:
    def is_active(p: rtc.RemoteParticipant) -> bool:
        return not (p.identity or "").startswith(OBSERVER_PREFIX)

    for p in ctx.room.remote_participants.values():
        if is_active(p):
            return p

    event = asyncio.Event()
    result: list[rtc.RemoteParticipant] = []

    def on_joined(p: rtc.RemoteParticipant) -> None:
        if is_active(p):
            result.append(p)
            event.set()

    ctx.room.on("participant_connected", on_joined)
    try:
        await asyncio.wait_for(event.wait(), timeout=PARTICIPANT_WAIT_TIMEOUT)
        return result[0] if result else None
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for participant.")
        return None
    finally:
        ctx.room.off("participant_connected", on_joined)


if __name__ == "__main__":
    cli.run_app(server)
