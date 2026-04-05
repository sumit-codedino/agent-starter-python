import asyncio
import logging
import os

from livekit import api
from livekit.agents import Agent

from backend import BackendClient, UserState

logger = logging.getLogger("stages.base")


class LoanStageAgent(Agent):
    """
    Base class for loan lifecycle stage agents.

    Provides:
    - user_state: borrower context from room metadata
    - backend: client for reporting outcomes
    - _end_call(message): speak goodbye then hang up
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

    def _end_call(self, message: str = "") -> None:
        """
        Speak the goodbye message, then delete the room to hang up.
        No timers — session.say() awaits TTS completion, then room deletion
        disconnects the SIP caller immediately.
        """
        async def _shutdown():
            room_name = self.session.room_io.room.name

            if message:
                await self.session.say(message, allow_interruptions=False)

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
