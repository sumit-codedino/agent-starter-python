import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import aiohttp

logger = logging.getLogger("backend")

_http_session: Optional[aiohttp.ClientSession] = None


async def _get_http() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session


@dataclass
class UserState:
    user_id: str           # maps to borrower_id
    current_stage: str
    name: str
    phone: str
    # Lead context
    ref_source: str = "other"
    loan_amount_interest: int = 150000
    city: str = ""
    # Lifecycle
    loan_data: dict = field(default_factory=dict)
    call_history: list[dict] = field(default_factory=list)
    missed_emi_count: int = 0
    # Cold call tracking
    cold_call_attempts: int = 0
    borrower_need: Optional[str] = None


@dataclass
class TransitionResult:
    next_stage: str
    allowed: bool
    reason: Optional[str] = None


class BackendClient:
    """
    Single interface for all backend communication.
    The backend is the source of truth for user state and stage transitions.
    """

    def __init__(self) -> None:
        self.api_url = os.getenv("API_URL", "")

    async def get_user_state(self, user_id: str) -> Optional[UserState]:
        """Fetch current user state at call start."""
        try:
            http = await _get_http()
            async with http.get(
                f"{self.api_url}/users/{user_id}/state",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.error(f"get_user_state failed: {resp.status}")
                    return None
                data = await resp.json()
                return UserState(
                    user_id=data["user_id"],
                    current_stage=data["current_stage"],
                    name=data["name"],
                    phone=data["phone"],
                    ref_source=data.get("ref_source", "other"),
                    loan_amount_interest=data.get("loan_amount_interest", 150000),
                    city=data.get("city", ""),
                    loan_data=data.get("loan_data", {}),
                    call_history=data.get("call_history", []),
                    missed_emi_count=data.get("missed_emi_count", 0),
                    cold_call_attempts=data.get("cold_call_attempts", 0),
                    borrower_need=data.get("borrower_need"),
                )
        except Exception as e:
            logger.error(f"get_user_state error: {e}", exc_info=True)
            return None

    async def request_transition(
        self,
        user_id: str,
        from_stage: str,
        to_stage: str,
        context: dict | None = None,
    ) -> TransitionResult:
        """Validate and execute a stage transition."""
        try:
            http = await _get_http()
            async with http.post(
                f"{self.api_url}/users/{user_id}/transition",
                json={"from_stage": from_stage, "to_stage": to_stage, "context": context or {}},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                return TransitionResult(
                    next_stage=data.get("next_stage", from_stage),
                    allowed=data.get("allowed", False),
                    reason=data.get("reason"),
                )
        except Exception as e:
            logger.error(f"request_transition error: {e}", exc_info=True)
            return TransitionResult(
                next_stage=from_stage,
                allowed=False,
                reason="Could not reach backend. Please try again.",
            )

    async def report_call_outcome(
        self,
        user_id: str,
        outcome: Literal["hot_lead", "callback", "not_interested", "no_response"],
        callback_time: Optional[str] = None,
        borrower_need: Optional[str] = None,
    ) -> None:
        """
        Report the outcome of a cold call.
        Backend decides next action:
          hot_lead       → schedules Stage 02 outbound call
          callback       → schedules retry at callback_time
          not_interested → marks declined
          no_response    → schedules retry in 4 hours (max 3 attempts)
        """
        try:
            http = await _get_http()
            await http.post(
                f"{self.api_url}/users/{user_id}/call-outcome",
                json={
                    "outcome": outcome,
                    "callback_time": callback_time,
                    "borrower_need": borrower_need,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            )
            logger.info(f"[{user_id}] Call outcome reported: {outcome}")
        except Exception as e:
            logger.error(f"report_call_outcome error: {e}", exc_info=True)

    async def save_call_summary(self, user_id: str, stage: str, summary: str) -> None:
        """Persist a per-call summary for context in future calls."""
        try:
            http = await _get_http()
            await http.post(
                f"{self.api_url}/users/{user_id}/call-summary",
                json={"stage": stage, "summary": summary},
                timeout=aiohttp.ClientTimeout(total=5),
            )
        except Exception as e:
            logger.error(f"save_call_summary error: {e}", exc_info=True)

    async def end_call(self, room_name: str, reason: str = "") -> None:
        """Signal backend to end the call and delete the LiveKit room."""
        try:
            http = await _get_http()
            await http.post(
                f"{self.api_url}/calls/end",
                json={"room_name": room_name, "reason": reason},
                timeout=aiohttp.ClientTimeout(total=5),
            )
        except Exception as e:
            logger.error(f"end_call error: {e}", exc_info=True)

    async def flag_human_handoff(self, user_id: str, reason: str) -> None:
        """Flag user for human agent handoff."""
        try:
            http = await _get_http()
            await http.post(
                f"{self.api_url}/users/{user_id}/handoff",
                json={"reason": reason},
                timeout=aiohttp.ClientTimeout(total=5),
            )
        except Exception as e:
            logger.error(f"flag_human_handoff error: {e}", exc_info=True)
