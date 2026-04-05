import logging
from typing import Optional

from backend import BackendClient, UserState

logger = logging.getLogger("stages")


def stage_to_agent(
    stage_id: str,
    user_state: UserState,
    backend: BackendClient,
) -> Optional[object]:
    from stages.s01_cold_call import ColdCallAgent
    from stages.s02_offer_presentation import OfferPresentationAgent

    registry: dict[str, type] = {
        "cold_call": ColdCallAgent,
        "offer_presentation": OfferPresentationAgent,
    }

    agent_class = registry.get(stage_id)
    if agent_class is None:
        logger.warning(f"No agent registered for stage: '{stage_id}'")
        return None

    return agent_class(user_state=user_state, backend=backend)
