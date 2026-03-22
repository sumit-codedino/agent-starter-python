import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional

import aiohttp
import boto3
from botocore.exceptions import ClientError
from openai import AsyncOpenAI
from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    metrics,
)
from livekit.agents.voice.room_io import RoomInputOptions, TextInputEvent
from livekit.agents.voice import CloseEvent, MetricsCollectedEvent
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel


logger = logging.getLogger("agent")
load_dotenv(".env.local")

# Constants
OBSERVER_PREFIX = "observer_"
PARTICIPANT_WAIT_TIMEOUT = 300.0  # 5 minutes
INACTIVE_PARTICIPANT_TIMEOUT = 300.0  # 5 minutes - timeout to close session if no active participants
DEFAULT_VOICE_ID = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"

# S3 Configuration
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Shared clients (created once, reused across calls)
_s3_client = None
_openai_client = None
_http_session: Optional[aiohttp.ClientSession] = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=S3_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    return _s3_client


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


async def _get_http_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session

async def upload_to_s3(content: str, s3_key: str, content_type: str = "application/json") -> Optional[str]:
    """Upload content to S3 bucket.
    
    Args:
        content: The content to upload (string)
        s3_key: The S3 key (path) where the file will be stored
        content_type: The content type of the file
        
    Returns:
        S3 URL if successful, None otherwise
    """
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not configured, skipping upload")
        return None
    
    try:
        s3_client = _get_s3_client()

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=content.encode("utf-8"),
            ContentType=content_type,
        )
        
        s3_url = f"s3://{S3_BUCKET}/{s3_key}"
        logger.info(f"Successfully uploaded to S3: {s3_url}")
        return s3_url
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error uploading to S3: {e}", exc_info=True)
        return None


async def generate_conversation_summary_from_transcript(transcript_json: str) -> str:
    try:
        transcript_data = json.loads(transcript_json)
        conversation_text = json.dumps(
            transcript_data.get("conversation_history", {}),
            indent=2
        )

        client = _get_openai_client()

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates concise summaries of conversations."
                },
                {
                    "role": "user",
                    "content": f"Please provide a summary of this conversation:\n\n{conversation_text}"
                }
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Failed to generate summary: {e}", exc_info=True)
        return "Summary generation failed"



async def get_agent_details(agent_id: str):
    api_url = os.getenv('API_URL')
    if not api_url:
        logger.warning("API_URL not configured, skipping agent details fetch")
        return None
    
    try:
        http = await _get_http_session()
        async with http.post(
            f"{api_url}/get-agent",
            json={"agent_id": agent_id}
        ) as response:
            if response.status != 200:
                logger.error(f"Failed to get agent details: {response.status}")
                return None
            data = await response.json()
            logger.info(f"Agent details: {data}")
            return data
    except Exception as e:
        logger.error(f"Error getting agent details: {e}", exc_info=True)
        return None


class Assistant(Agent):
    def __init__(
        self,
        system_prompt: str,
        agent_id: str | None = None,
        welcome_message: str | None = None,
        welcome_message_enabled: bool = False,
    ) -> None:
        super().__init__(instructions=system_prompt)
        self.agent_id = agent_id
        self.welcome_message = welcome_message
        self.welcome_message_enabled = welcome_message_enabled

    @function_tool()
    async def query_documents(self, context: RunContext, query: str) -> str:
        """Search the knowledge base to answer user questions.

        Use this tool whenever the user asks something that could be answered
        by documents, policies, product info, or any stored knowledge.
        Do NOT use this tool for general knowledge questions.

        Args:
            query: The user's question or topic to search for.
        """
        api_url = os.getenv("API_URL")
        if not api_url:
            return "Knowledge base is not configured."

        if not self.agent_id:
            logger.warning("query_documents called but agent_id is not set")

        try:
            http = await _get_http_session()
            async with http.post(
                f"{api_url}/query-documents",
                json={"query": query, "agent_id": self.agent_id},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status != 200:
                    logger.error(f"query_documents API returned {response.status}")
                    return "Failed to search the knowledge base."

                data = await response.json()
                results = data.get("results", [])

                # Filter by relevance score and take top 3
                relevant = [r for r in results if r.get("score", 0) >= 0.60][:3]

                if not relevant:
                    return "No relevant information found in the knowledge base."

                # Cap each chunk to avoid bloating LLM context
                MAX_CHUNK_CHARS = 600
                formatted = "\n\n".join(
                    f"[{r['document_name']}]: {r['text'][:MAX_CHUNK_CHARS]}"
                    for r in relevant
                )
                return formatted

        except asyncio.TimeoutError:
            logger.error("query_documents timed out after 5s")
            return "Knowledge base search timed out."
        except Exception as e:
            logger.error(f"Error querying documents: {e}", exc_info=True)
            return "Failed to search the knowledge base."

    async def on_enter(self):
        logger.info("=== on_enter called ===")
        
        # Only send welcome message if enabled and message is provided
        if self.welcome_message_enabled and self.welcome_message:
            try:
                logger.info(f"Sending welcome message: {self.welcome_message}")
                await self.session.say(
                    self.welcome_message,
                    allow_interruptions=True,
                )
                logger.info("Welcome message sent successfully")
            except Exception as e:
                logger.error(f"Error sending welcome message: {e}", exc_info=True)
        else:
            logger.info("Welcome message disabled or not provided, skipping greeting")


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Connect to the room first to ensure room metadata is populated
    await ctx.connect()
    logger.info("Connected to room")
    
    # Now access room metadata - it should be populated after connecting
    metadata = json.loads(ctx.room.metadata or "{}")
    logger.info(f"Room metadata: {metadata}")
    logger.info(f"Room name: {ctx.room.name}")

    agent_id = metadata.get("agent_id") 
    logger.info(f"Agent id: {agent_id}")

    agent_language = metadata.get("language", "English")
    agent_language_code = metadata.get("language_code", "en")

    # Default prompt if agent details cannot be fetched
    default_prompt = "You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text. You eagerly assist users with their questions by providing information from your extensive knowledge. Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols. You are curious, friendly, and have a sense of humor."
    
    # Initialize agent configuration variables with defaults
    system_prompt = default_prompt
    welcome_message = None
    welcome_message_enabled = False
    agent_voice_id = None
    agent_language = "English"
    agent_language_code = "en"
    
    if not agent_id:
        logger.warning("No agent_id in metadata, using default prompt")
    else:
        agent_details_response = await get_agent_details(agent_id)
        
        if agent_details_response and agent_details_response.get("agent"):
            agent_details = agent_details_response.get("agent")
            system_prompt = agent_details.get("agent_prompt") or default_prompt
            welcome_message = agent_details.get("agent_welcome_message", "")
            welcome_message_enabled = agent_details.get("agent_welcome_message_enabled", False)
            agent_voice_id = agent_details.get("agent_voice_id")
            
            if agent_voice_id:
                logger.info(f"Using voice ID from API for agent: {agent_voice_id}")
            else:
                logger.warning("No agent_voice_id in agent details, using default voice")
            
            if agent_language_code and agent_language_code != "en":
                logger.info(f"Using language from API for agent: {agent_language} ({agent_language_code})")
            else:
                logger.info(f"Using default language: English (en)")
            
            if system_prompt != default_prompt:
                logger.info(f"Using system prompt from API for agent: {agent_id}")
            else:
                logger.warning("No agent_prompt in agent details, using default prompt")
            
            if welcome_message_enabled and welcome_message:
                logger.info(f"Welcome message enabled: {welcome_message[:50]}...")
            elif welcome_message_enabled:
                logger.warning("Welcome message enabled but no message provided")
        else:
            logger.warning("Failed to get agent details or invalid response, using default prompt")
    
    logger.info(f"Selected Agent Language: {agent_language} ({agent_language_code})")
    if agent_language_code != "en":
        system_prompt += f"\n\nAlways respond in {agent_language}."

    # Create agent instance
    agent = Assistant(
        system_prompt=system_prompt,
        agent_id=agent_id,
        welcome_message=welcome_message,
        welcome_message_enabled=welcome_message_enabled,
    )

    def is_observer_participant(participant) -> bool:
        """Check if a participant is an observer."""
        return (
            hasattr(participant, 'identity') 
            and participant.identity 
            and participant.identity.startswith(OBSERVER_PREFIX)
        )
    
    def can_participant_publish(participant) -> bool:
        """Check if a participant can publish (not an observer)."""
        return not is_observer_participant(participant)
    
    def find_active_participant() -> Optional[rtc.RemoteParticipant]:
        """Find the first active (non-observer) participant in the room."""
        for participant in ctx.room.remote_participants.values():
            if can_participant_publish(participant):
                return participant
        return None
    
    async def update_room_status(room_name: str, room_status: str) -> bool:
        """Update the room status in the database (async)."""
        api_url = os.getenv('API_URL')
        if not api_url:
            logger.warning("API_URL not configured, skipping room status update")
            return False
        
        try:
            http = await _get_http_session()
            async with http.post(
                f"{api_url}/update-room",
                json={"room_name": room_name, "new_status": room_status}
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to update room status: {response.status}")
                    return False
                return True
        except Exception as e:
            logger.error(f"Error updating room status: {e}", exc_info=True)
            return False

    async def wait_for_active_participant() -> Optional[rtc.RemoteParticipant]:
        """Wait for the first non-observer participant to join."""
        # Check existing participants first
        participant = find_active_participant()
        if participant:
            logger.info(f"Found active participant: {participant.identity}")
            await update_room_status(ctx.room.name, "active")
            return participant
        
        # Wait for one to join
        logger.info("Waiting for a non-observer participant to join...")
        participant_event = asyncio.Event()
        active_participant = None
        
        def on_participant_connected(p: rtc.RemoteParticipant):
            nonlocal active_participant
            if can_participant_publish(p):
                logger.info(f"Active participant joined: {p.identity}")
                active_participant = p
                participant_event.set()
            else:
                logger.debug(f"Observer joined: {p.identity} (skipping for audio)")
        
        ctx.room.on("participant_connected", on_participant_connected)
        
        try:
            await asyncio.wait_for(participant_event.wait(), timeout=PARTICIPANT_WAIT_TIMEOUT)
            if active_participant:
                await update_room_status(ctx.room.name, "active")
            return active_participant
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for active participant.")
            participant = find_active_participant()
            if participant:
                await update_room_status(ctx.room.name, "active")
            return participant
        finally:
            ctx.room.off("participant_connected", on_participant_connected)
    
    # Wait for active participant
    active_participant = await wait_for_active_participant()
    if not active_participant:
        logger.warning("No active participant found. Agent will not start.")
        return
    
    logger.info(f"Starting agent session for active participant: {active_participant.identity}")

    # Initialize egress recording
    egress_id: Optional[str] = None
    lkapi = api.LiveKitAPI()
    
    async def start_egress_recording():
        """Start egress video recording to S3."""
        nonlocal egress_id
        try:
            if not S3_BUCKET:
                logger.warning("S3_BUCKET not configured, skipping egress recording")
                return
            
            room_name = ctx.room.name
            filepath = f"media/{room_name}/recording.mp4"
            
            req = api.RoomCompositeEgressRequest(
                room_name=room_name,
                layout="speaker",
                file_outputs=[
                    api.EncodedFileOutput(
                        file_type=api.EncodedFileType.MP4,
                        filepath=filepath,
                        s3=api.S3Upload(
                            bucket=S3_BUCKET,
                            region=S3_REGION,
                            access_key=AWS_ACCESS_KEY_ID,
                            secret=AWS_SECRET_ACCESS_KEY,
                        ),
                    )
                ],
            )
            
            egress_info = await lkapi.egress.start_room_composite_egress(req)
            egress_id = egress_info.egress_id
            logger.info(f"Egress recording started. Egress ID: {egress_id}, File: {filepath}")
        except Exception as e:
            logger.error(f"Failed to start egress recording: {e}", exc_info=True)
    
    async def upload_transcript_and_summary():
        """Upload transcript and summary to S3 when session ends."""
        nonlocal transcript_uploaded
        if transcript_uploaded:
            logger.debug("Transcript already uploaded, skipping duplicate upload")
            return
            
        if not session:
            logger.warning("No session available for transcript upload")
            return
        
        transcript_uploaded = True  # Set flag before upload to prevent race conditions
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            room_name = ctx.room.name
            
            # 1. Upload full transcript
            transcript_data = {
                "room_name": room_name,
                "timestamp": timestamp,
                "conversation_history": session.history.to_dict(),
            }
            transcript_json = json.dumps(transcript_data, indent=2)
            transcript_key = f"media/{room_name}/transcript.json"
            transcript_url = await upload_to_s3(transcript_json, transcript_key)
            
            if transcript_url:
                logger.info(f"Transcript uploaded to S3: {transcript_url}")
            
            # 2. Generate summary from the uploaded transcript JSON and upload
            try:
                # Use the transcript JSON we just created instead of session.history
                summary_text = await generate_conversation_summary_from_transcript(transcript_json)
                summary_data = {
                    "room_name": room_name,
                    "timestamp": timestamp,
                    "summary": summary_text,
                    "transcript_url": transcript_url,
                    "egress_id": egress_id,
                }
                summary_json = json.dumps(summary_data, indent=2)
                summary_key = f"media/{room_name}/summary.json"
                summary_url = await upload_to_s3(summary_json, summary_key)
                
                if summary_url:
                    logger.info(f"Summary uploaded to S3: {summary_url}")
            except Exception as e:
                logger.error(f"Failed to generate/upload summary: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Failed to upload transcript/summary: {e}", exc_info=True)
    

    usage_collector = metrics.UsageCollector()
    current_linked_participant = active_participant
    session: Optional[AgentSession] = None
    text_stream_tasks = []  # Track tasks for cleanup
    transcript_uploaded = False  # Flag to prevent duplicate uploads
    participant_monitor_task: Optional[asyncio.Task] = None

        # Unified text message handler
    async def handle_text_message(message: str, participant_identity: str, source: str = "stream"):
        """Handle text messages from any participant."""
        if not session:
            return
        
        logger.info(f"Received text message from {participant_identity} ({source}): {message}")
        try:
            session.interrupt()
            session.generate_reply(user_input=message)
        except Exception as e:
            logger.error(f"Error processing text message: {e}", exc_info=True)
    
    # Text input handler for linked participant
    def custom_text_input_handler(_: AgentSession, event: TextInputEvent) -> None:
        """Handle text from linked participant."""
        task = asyncio.create_task(
            handle_text_message(event.text, event.participant_identity, "linked")
        )
        text_stream_tasks.append(task)
    
    # Text stream handler for all participants (including observers)
    def handle_text_stream(reader, participant_identity: str):
        """Handle text streams from any participant on lk.chat topic."""
        async def read_stream():
            try:
                message = await reader.read_all()
                await handle_text_message(message, participant_identity, "stream")
            except Exception as e:
                logger.error(f"Error reading text stream: {e}", exc_info=True)
        
        task = asyncio.create_task(read_stream())
        text_stream_tasks.append(task)
    
    ctx.room.register_text_stream_handler("lk.chat", handle_text_stream)
    
    async def cleanup_session_resources():
        """Cleanup session resources (transcript upload, recording stop, etc.)."""
        nonlocal transcript_uploaded, egress_id, text_stream_tasks, current_linked_participant
        
        logger.info("Cleaning up session resources...")
        
        # Cancel and cleanup text stream tasks
        for task in text_stream_tasks:
            if not task.done():
                task.cancel()
        text_stream_tasks.clear()
        
        # Upload transcript and summary
        await upload_transcript_and_summary()
        
        # Stop egress recording if running
        if egress_id:
            try:
                logger.info(f"Stopping egress recording. Egress ID: {egress_id}")
                stop_request = api.StopEgressRequest(egress_id=egress_id)
                await lkapi.egress.stop_egress(stop_request)
                logger.info(f"Egress recording stopped. Egress ID: {egress_id}")
            except Exception as e:
                logger.error(f"Failed to stop egress recording: {e}", exc_info=True)
        
        current_linked_participant = None
    
    async def monitor_participants():
        """Monitor for inactive participants and close session if timeout exceeded."""
        nonlocal session, current_linked_participant  # noqa: F841
        
        no_participant_start_time = None
        check_interval = 30  # Check every 30 seconds
        
        while session is not None:
            try:
                await asyncio.sleep(check_interval)
                
                # Check if there are any active participants
                active_participant = find_active_participant()
                
                if not active_participant:
                    # No active participants found
                    if current_linked_participant is None:
                        # Already cleaned up, exit
                        break
                    
                    # Track when we first detected no participants
                    if no_participant_start_time is None:
                        no_participant_start_time = datetime.now()
                        logger.warning("No active participants found. Starting timeout countdown...")
                    else:
                        # Check if timeout has been exceeded
                        elapsed = (datetime.now() - no_participant_start_time).total_seconds()
                        remaining = INACTIVE_PARTICIPANT_TIMEOUT - elapsed
                        
                        if remaining <= 0:
                            # Timeout exceeded, close session
                            logger.info(f"No active participants for {INACTIVE_PARTICIPANT_TIMEOUT}s. Closing session...")
                            await cleanup_session_resources()
                            if session is not None:
                                try:
                                    await session.aclose()
                                except Exception as e:
                                    logger.warning(f"Error closing session: {e}")
                                finally:
                                    session = None
                            break
                        elif remaining <= 60:
                            # Log warning when less than 1 minute remaining
                            logger.warning(f"No active participants. Will close in {int(remaining)}s if none join.")
                else:
                    # Active participant found
                    if no_participant_start_time is not None:
                        logger.info(f"Active participant rejoined: {active_participant.identity}. Resetting timeout.")
                        no_participant_start_time = None
                    
                    # Update current linked participant if changed
                    if active_participant.identity != (current_linked_participant.identity if current_linked_participant else None):
                        logger.info(f"Active participant changed to: {active_participant.identity}")
                        current_linked_participant = active_participant
                        
            except asyncio.CancelledError:
                logger.info("Participant monitor task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in participant monitor: {e}", exc_info=True)
                await asyncio.sleep(check_interval)  # Wait before retrying
    
    # Session management
    async def start_session_with_participant(participant: rtc.RemoteParticipant):
        """Start or restart the agent session with a specific participant."""
        nonlocal session, current_linked_participant, egress_id, participant_monitor_task, agent_voice_id
        
        # Close existing session
        if session is not None:
            try:
                await session.aclose()
            except Exception as e:
                logger.warning(f"Error closing previous session: {e}")
        
        # Cancel existing monitor task
        if participant_monitor_task and not participant_monitor_task.done():
            participant_monitor_task.cancel()
            try:
                await participant_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Create new session with agent configuration
        session = AgentSession(
            stt=inference.STT(model="cartesia/ink-whisper", language=agent_language_code),
            llm=inference.LLM(model="openai/gpt-4.1-mini"),
            tts=inference.TTS(
                model="cartesia/sonic-3",
                voice=agent_voice_id if agent_voice_id else DEFAULT_VOICE_ID,
                language=agent_language_code
            ),
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata["vad"],
            preemptive_generation=True,
        )
        
        # Register metrics collector
        @session.on("metrics_collected")
        def _on_metrics_collected(ev: MetricsCollectedEvent):
            metrics.log_metrics(ev.metrics)
            usage_collector.collect(ev.metrics)
        
        # Register close event handler
        @session.on("close")
        def _on_session_close(ev: CloseEvent):
            """Handle session close event."""
            nonlocal participant_monitor_task
            error_msg = str(ev.error) if ev.error else None
            logger.info(f"Session closed. Error: {error_msg if error_msg else 'None'}")
            
            # Cancel participant monitor task
            if participant_monitor_task and not participant_monitor_task.done():
                participant_monitor_task.cancel()
            
        
        current_linked_participant = participant
        
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                participant_identity=participant.identity,
                audio_enabled=True,
                noise_cancellation=noise_cancellation.BVC(),
                text_enabled=True,
                text_input_cb=custom_text_input_handler,
                close_on_disconnect=False,
            ),
        )
        
        logger.info(f"Agent session started/restarted. Audio linked to: {participant.identity}")
        
        # Start egress recording when session starts
        await start_egress_recording()
        
        # Start participant monitoring task
        participant_monitor_task = asyncio.create_task(monitor_participants())
    
    
    # Setup event handlers
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    async def cleanup_on_shutdown():
        """Cleanup function called on agent shutdown."""
        nonlocal session, egress_id, text_stream_tasks
        # Cancel and cleanup text stream tasks
        for task in text_stream_tasks:
            if not task.done():
                task.cancel()
        # Wait for tasks to complete cancellation
        if text_stream_tasks:
            await asyncio.gather(*text_stream_tasks, return_exceptions=True)
        text_stream_tasks.clear()
        
        
        # Close LiveKit API connection (last, after all operations)
        try:
            await lkapi.aclose()
        except Exception as e:
            logger.warning(f"Error closing LiveKit API on shutdown: {e}")
    
    # Handle participant disconnection (update current_linked_participant, but don't close session)
    def handle_participant_disconnected(participant: rtc.RemoteParticipant):
        """Handle participant disconnection - update current linked participant."""
        nonlocal current_linked_participant
        if current_linked_participant and participant.identity == current_linked_participant.identity:
            logger.info(f"Linked participant {participant.identity} disconnected. Monitor will check for timeout.")
            current_linked_participant = None
    
    ctx.room.on("participant_disconnected", handle_participant_disconnected)
    ctx.add_shutdown_callback(cleanup_session_resources)

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(cleanup_on_shutdown)
    
    # Start initial session
    await start_session_with_participant(active_participant)
    logger.info("Text messages can be received from all participants via lk.chat topic")
    logger.info(f"Session will auto-close if no active participants for {INACTIVE_PARTICIPANT_TIMEOUT}s")


if __name__ == "__main__":
    cli.run_app(server)
