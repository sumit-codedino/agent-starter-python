import asyncio
import logging
import os
from typing import TYPE_CHECKING

from livekit import api
from livekit.agents import Agent, RunContext, function_tool

from backend import BackendClient, UserState

if TYPE_CHECKING:
    pass

logger = logging.getLogger("stages.base")


class LoanStageAgent(Agent):
    """
    Base class for all loan lifecycle stage agents.

    Each stage agent inherits this and gets:
    - user_state: full context loaded at call start
    - backend: shared client for state/transition calls
    - transition(): validated stage switching via backend

    Stage agents should NOT call session.update_agent() directly.
    Always go through self.transition() so the backend stays authoritative.
    """

    def __init__(
        self,
        *,
        instructions: str,
        user_state: UserState,
        backend: BackendClient,
        stage_id: str,
    ) -> None:
        super().__init__(instructions=instructions)
        self.user_state = user_state
        self.backend = backend
        self.stage_id = stage_id

    async def transition(
        self,
        context: RunContext,
        to_stage: str,
        transition_context: dict | None = None,
    ) -> str:
        """
        Request a stage transition from the backend.
        If approved, swaps the active agent. Returns a message for the LLM to speak.
        If denied, returns an explanation the LLM can relay to the user.
        """
        result = await self.backend.request_transition(
            user_id=self.user_state.user_id,
            from_stage=self.stage_id,
            to_stage=to_stage,
            context=transition_context,
        )

        if not result.allowed:
            logger.warning(
                f"Transition denied: {self.stage_id} → {to_stage}. Reason: {result.reason}"
            )
            return result.reason or "That step isn't available right now."

        logger.info(f"Transitioning: {self.stage_id} → {result.next_stage}")

        # Import here to avoid circular imports at module load time
        from stages import stage_to_agent

        next_agent = stage_to_agent(result.next_stage, self.user_state, self.backend)
        if next_agent is None:
            logger.error(f"No agent registered for stage: {result.next_stage}")
            return "There was an error moving to the next step. Please call back."

        self.session.update_agent(next_agent)
        return ""  # on_enter of the next agent handles the greeting

    def _end_call(self) -> None:
        """
        Schedule graceful call teardown after the current TTS response finishes.
        Drains speech first, then deletes the room to disconnect the SIP caller.
        Called from outcome tools — fire-and-forget via create_task.
        """
        async def _shutdown():
            # Capture room name before session state is cleaned up
            room_name = self.session.room_io.room.name

            # Wait for session to fully close (drain=True lets TTS finish first)
            close_event = asyncio.Event()
            self.session.on("close")(lambda _: close_event.set())
            self.session.shutdown(drain=True)
            try:
                await asyncio.wait_for(close_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning(f"Session close timed out for room {room_name}")

            # Now delete the room — sends SIP BYE to Twilio, hangs up customer
            lkapi = api.LiveKitAPI(
                url=os.getenv("LIVEKIT_URL", ""),
                api_key=os.getenv("LIVEKIT_API_KEY", ""),
                api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            )
            try:
                await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
                logger.info(f"Room deleted: {room_name}")
            except Exception as e:
                logger.error(f"Failed to delete room {room_name}: {e}")
            finally:
                await lkapi.aclose()

        asyncio.create_task(_shutdown())

    def _last_call_summary(self) -> str:
        """Return the most recent call summary, or empty string if none."""
        if not self.user_state.call_history:
            return ""
        return self.user_state.call_history[-1].get("summary", "")

    def _loan_context(self) -> str:
        """Formatted loan details for injection into system prompts."""
        d = self.user_state.loan_data
        if not d:
            return ""
        lines = []
        if d.get("amount"):
            lines.append(f"Loan amount: ₹{d['amount']:,}")
        if d.get("tenure_months"):
            lines.append(f"Tenure: {d['tenure_months']} months")
        if d.get("interest_rate"):
            lines.append(f"Interest rate: {d['interest_rate']}% p.a.")
        if d.get("emi"):
            lines.append(f"Monthly EMI: ₹{d['emi']:,}")
        return "\n".join(lines)
