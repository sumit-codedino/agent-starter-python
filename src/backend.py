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
    user_id: str
    current_stage: str
    name: str
    phone: str
    ref_source: str = "other"
    loan_amount_interest: int = 150000
    city: str = ""
    cold_call_attempts: int = 0
    borrower_need: Optional[str] = None
    borrower_mood: Optional[str] = None
    # Stage 02 — loan terms (populated by backend before offer call)
    loan_terms: dict = field(default_factory=dict)
    # Expected keys: roi_annual_pct, tenure_months, emi_amount,
    #                total_repayable, processing_fee, net_disbursement


class BackendClient:
    """Interface for backend communication. Backend is source of truth for outcomes."""

    def __init__(self) -> None:
        self.api_url = os.getenv("API_URL", "")
        self.conversation_id: str = ""
        self.stage: str = ""

    async def get_prompt_data(self, stage_id: str) -> Optional[dict]:
        """
        Fetch the prompt data (template + first_message) for a stage from the backend.
        Returns None on error or if no record is stored — callers should fall back
        to hardcoded defaults in that case.
        """
        try:
            http = await _get_http()
            async with http.get(
                f"{self.api_url}/prompts",
                params={"stage_id": stage_id},
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status == 404:
                    logger.info(f"No DynamoDB prompt for stage '{stage_id}', using hardcoded defaults")
                    return None
                if resp.status != 200:
                    logger.warning(f"get_prompt_data HTTP {resp.status} for '{stage_id}'")
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"get_prompt_data error for '{stage_id}': {e}")
            return None

    async def report_call_outcome(
        self,
        user_id: str,
        outcome: str,
        callback_time: Optional[str] = None,
        callback_date: Optional[str] = None,
        follow_up_time: Optional[str] = None,
        borrower_need: Optional[str] = None,
        borrower_mood: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        objection_detail: Optional[str] = None,
        any_objection_raised: bool = False,
        stage_03_context_note: Optional[str] = None,
    ) -> None:
        """Report the outcome of any stage call to the backend."""
        try:
            http = await _get_http()
            await http.post(
                f"{self.api_url}/users/{user_id}/call-outcome",
                json={
                    "conversation_id": self.conversation_id,
                    "stage": self.stage,
                    "outcome": outcome,
                    "callback_time": callback_time,
                    "callback_date": callback_date,
                    "follow_up_time": follow_up_time,
                    "borrower_need": borrower_need,
                    "borrower_mood": borrower_mood,
                    "rejection_reason": rejection_reason,
                    "objection_detail": objection_detail,
                    "any_objection_raised": any_objection_raised,
                    "stage_03_context_note": stage_03_context_note,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            )
            logger.info(f"[{user_id}] Call outcome reported: {outcome}")
        except Exception as e:
            logger.error(f"report_call_outcome error: {e}", exc_info=True)
