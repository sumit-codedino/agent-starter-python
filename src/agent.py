import asyncio
import json
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess, TurnHandlingOptions, cli, inference, metrics
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from backend import BackendClient, UserState
from stages import stage_to_agent

load_dotenv(".env.local")

logger = logging.getLogger("loan-agent")

MAX_DURATION = {
    "cold_call": 180,
    "offer_presentation": 300,
}

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="loan-lifecycle-agent")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")

    # --- 1. Read metadata ---
    metadata = json.loads(ctx.room.metadata or "{}")
    user_id = metadata.get("user_id")
    if not user_id:
        logger.error("No user_id in room metadata.")
        return

    # --- 2. Build user state from room metadata ---
    user_state = UserState(
        user_id=user_id,
        current_stage=metadata.get("current_stage", "cold_call"),
        name=metadata.get("name", ""),
        phone=metadata.get("phone", ""),
        ref_source=metadata.get("ref_source", "other"),
        loan_amount_interest=metadata.get("loan_amount_interest", 150000),
        city=metadata.get("city", ""),
        cold_call_attempts=metadata.get("cold_call_attempts", 0),
        borrower_need=metadata.get("borrower_need"),
        borrower_mood=metadata.get("borrower_mood"),
        loan_terms=metadata.get("loan_terms") or {},
    )

    stage = user_state.current_stage
    logger.info(f"Stage '{stage}' | user={user_id}")

    # --- 3. Resolve agent ---
    backend = BackendClient()
    backend.conversation_id = metadata.get("conversation_id", "")
    backend.stage = stage
    prompt_data = await backend.get_prompt_data(stage)
    template = prompt_data.get("template") or None if prompt_data else None
    first_message = prompt_data.get("first_message") or None if prompt_data else None
    logger.info(f"Using template: {template}")
    logger.info(f"Using first_message: {first_message}")

    agent = stage_to_agent(stage, user_state, backend, template=template, first_message=first_message)
    if agent is None:
        logger.error(f"No agent for stage '{stage}'")
        return

    # --- 4. Wait for SIP participant ---
    participant = await _wait_for_participant(ctx)
    if not participant:
        logger.warning("No participant joined.")
        return

    # --- 5. Build and start session ---
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general", language="multi"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="faf0731e-dfb9-4cfc-8119-259a79b27e12",
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

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            participant_identity=participant.identity,
            audio_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # --- 6. Max duration safety net ---
    max_seconds = MAX_DURATION.get(stage)
    if max_seconds:
        asyncio.create_task(_enforce_max_duration(session, ctx.room.name, max_seconds))

    async def _on_shutdown():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(_on_shutdown)
    logger.info(f"Session live | user={user_state.name} | stage={stage}")


async def _enforce_max_duration(session: AgentSession, room_name: str, seconds: int) -> None:
    """Hard cap on call duration. Says goodbye, then deletes the room."""
    await asyncio.sleep(seconds)
    logger.info(f"Max call duration ({seconds}s) reached")
    try:
        await session.say("Theek hai — main baad mein call karti hoon. Dhanyawaad!", allow_interruptions=False)
    except Exception:
        pass

    from livekit import api
    import os
    lkapi = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL", ""),
        api_key=os.getenv("LIVEKIT_API_KEY", ""),
        api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
    )
    try:
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
    except Exception:
        pass
    finally:
        await lkapi.aclose()


async def _wait_for_participant(ctx: JobContext) -> rtc.RemoteParticipant | None:
    """Wait for the SIP participant (customer) to join the room."""
    for p in ctx.room.remote_participants.values():
        return p

    event = asyncio.Event()
    result: list[rtc.RemoteParticipant] = []

    def on_joined(p: rtc.RemoteParticipant) -> None:
        result.append(p)
        event.set()

    ctx.room.on("participant_connected", on_joined)
    try:
        await asyncio.wait_for(event.wait(), timeout=120.0)
        return result[0] if result else None
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for participant.")
        return None
    finally:
        ctx.room.off("participant_connected", on_joined)


if __name__ == "__main__":
    cli.run_app(server)
