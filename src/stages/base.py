import asyncio
import logging
from typing import TYPE_CHECKING

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
        Deletes the room, which disconnects the SIP caller.
        Called from outcome tools — fire-and-forget via create_task.
        """
        async def _shutdown():
            await self.session.shutdown(delete_room=True)

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
