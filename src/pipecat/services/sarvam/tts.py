#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam AI text-to-speech service implementation."""

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sarvam, you need to `pip install pipecat-ai[sarvam]`.")
    raise Exception(f"Missing module: {e}")


def language_to_sarvam_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Sarvam AI language codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Sarvam AI language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.BN: "bn-IN",  # Bengali
        Language.EN: "en-IN",  # English (India)
        Language.GU: "gu-IN",  # Gujarati
        Language.HI: "hi-IN",  # Hindi
        Language.KN: "kn-IN",  # Kannada
        Language.ML: "ml-IN",  # Malayalam
        Language.MR: "mr-IN",  # Marathi
        Language.OR: "od-IN",  # Odia
        Language.PA: "pa-IN",  # Punjabi
        Language.TA: "ta-IN",  # Tamil
        Language.TE: "te-IN",  # Telugu
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class SarvamTTSSettings(TTSSettings):
    """Settings for SarvamHttpTTSService and SarvamTTSService."""

    pass


class SarvamHttpTTSService(TTSService):
    """Text-to-Speech service using Sarvam AI's API.

    Converts text to speech using Sarvam AI's TTS models with support for multiple
    Indian languages. Provides control over voice characteristics like pitch, pace,
    and loudness.

    Example::

        tts = SarvamHttpTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            model="bulbul:v2",
            aiohttp_session=session,
            params=SarvamHttpTTSService.InputParams(
                language=Language.HI,
                pitch=0.1,
                pace=1.2
            )
        )

        # For bulbul v3 beta with any speaker:
        tts_v3 = SarvamHttpTTSService(
            api_key="your-api-key",
            voice_id="speaker_name",
            model="bulbul:v3,
            aiohttp_session=session,
            params=SarvamHttpTTSService.InputParams(
                language=Language.HI,
                temperature=0.8
            )
        )
    """

    Settings = SarvamTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for Sarvam TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English (India).
            pitch: Voice pitch adjustment (-0.75 to 0.75). Defaults to 0.0.
            pace: Speech pace multiplier (0.3 to 3.0). Defaults to 1.0.
            loudness: Volume multiplier (0.1 to 3.0). Defaults to 1.0.
            enable_preprocessing: Whether to enable text preprocessing. Defaults to False.
        """

        language: Optional[Language] = Language.EN
        pitch: Optional[float] = Field(default=0.0, ge=-0.75, le=0.75)
        pace: Optional[float] = Field(default=1.0, ge=0.3, le=3.0)
        loudness: Optional[float] = Field(default=1.0, ge=0.1, le=3.0)
        enable_preprocessing: Optional[bool] = False
        temperature: Optional[float] = Field(
            default=0.6,
            ge=0.01,
            le=1.0,
            description="Controls the randomness of the output for bulbul v3 beta. "
            "Lower values make the output more focused and deterministic, while "
            "higher values make it more random. Range: 0.01 to 1.0. Default: 0.6.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str = "anushka",
        model: str = "bulbul:v2",
        base_url: str = "https://api.sarvam.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Sarvam TTS service.

        Args:
            api_key: Sarvam AI API subscription key.
            aiohttp_session: Shared aiohttp session for making requests.
            voice_id: Speaker voice ID (e.g., "anushka", "meera"). Defaults to "anushka".
            model: TTS model to use ("bulbul:v2" or "bulbul:v3-beta" or "bulbul:v3"). Defaults to "bulbul:v2".
            base_url: Sarvam AI API base URL. Defaults to "https://api.sarvam.ai".
            sample_rate: Audio sample rate in Hz (8000, 16000, 22050, 24000). If None, uses default.
            params: Additional voice and preprocessing parameters. If None, uses defaults.
            settings: Runtime-updatable settings. When provided, values take precedence
                over ``voice_id`` and ``model``.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        params = params or SarvamHttpTTSService.InputParams()

        resolved_language = (
            language_to_sarvam_language(params.language) if params.language else "en-IN"
        )

        default_settings = self.Settings(
            model=model,
            voice=voice_id,
            language=resolved_language,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

        # Outgoing API request body — separate concern from framework Settings.
        self._request_payload: dict = {
            "language": default_settings.language,
            "enable_preprocessing": params.enable_preprocessing,
        }

        if default_settings.model in ("bulbul:v3-beta", "bulbul:v3"):
            self._request_payload.update(
                {
                    "temperature": getattr(params, "temperature", 0.6),
                    "model": default_settings.model,
                }
            )
        else:
            self._request_payload.update(
                {
                    "pitch": params.pitch,
                    "pace": params.pace,
                    "loudness": params.loudness,
                    "model": default_settings.model,
                }
            )

    @classmethod
    def get_voices(cls, api_key: str):
    # Language list (Sarvam Bulbul v3 supports all for every speaker)
        languages = [
        "hi-IN", "bn-IN", "ta-IN", "te-IN", "gu-IN",
        "kn-IN", "ml-IN", "mr-IN", "pa-IN", "od-IN", "en-IN"
        ]

    # Gender mapping
        male_voices = {
        "Shubh", "Aditya", "Rahul", "Rohan", "Amit", "Dev", "Ratan",
        "Varun", "Manan", "Sumit", "Kabir", "Aayan", "Ashutosh",
        "Advait", "Anand", "Tarun", "Sunny", "Mani", "Gokul",
        "Vijay", "Mohit", "Rehan", "Soham"
        }

        female_voices = {
        "Ritu", "Priya", "Neha", "Pooja", "Simran", "Kavya", "Ishita",
        "Shreya", "Roopa", "Amelia", "Sophia", "Tanya", "Shruti",
        "Suhani", "Kavitha", "Rupali"
        }

        speakers = list(male_voices | female_voices)

        voice_catalog = []

        for speaker in speakers:
            gender = (
                "male" if speaker in male_voices
                else "female" if speaker in female_voices
                else "unknown"
            )

            for lang in languages:
                voice_catalog.append({
                    "name": speaker,
                    "voice_id": f"sarvam-{speaker.lower()}-{lang}",
                    "description": None,
                    "gender": gender,
                    "language": lang,
                    "sample_url": None,
                    "accent": None,
                })

        return voice_catalog


    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def start(self, frame: StartFrame):
        """Start the Sarvam TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._request_payload["sample_rate"] = self.sample_rate

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Sarvam AI's API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics(context_id=context_id)

            payload = {
                **self._request_payload,
                "text": text,
                "target_language_code": self._request_payload["language"],
                "speaker": self._settings.voice,
                "sample_rate": self.sample_rate,
            }
            payload.pop("language", None)

            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json",
                **sdk_headers(),
            }

            url = f"{self._base_url}/text-to-speech"

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Sarvam API error: {error_text}")
                    return

                response_data = await response.json()

            await self.start_tts_usage_metrics(text)

            # Decode base64 audio data
            if "audios" not in response_data or not response_data["audios"]:
                yield ErrorFrame(error="No audio data received")
                return

            # Get the first audio (there should be only one for single text input)
            base64_audio = response_data["audios"][0]
            audio_data = base64.b64decode(base64_audio)

            # Strip WAV header (first 44 bytes) if present
            if audio_data.startswith(b"RIFF"):
                logger.debug("Stripping WAV header from Sarvam audio data")
                audio_data = audio_data[44:]

            await self.stop_ttfb_metrics(context_id=context_id)

            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}", exception=e)


class SarvamTTSService(InterruptibleTTSService):
    """WebSocket-based text-to-speech service using Sarvam AI.

    Provides streaming TTS with real-time audio generation for multiple Indian languages.
    Supports voice control parameters like pitch, pace, and loudness adjustment.

    Example::

        tts = SarvamTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            model="bulbul:v2",
            params=SarvamTTSService.InputParams(
                language=Language.HI,
                pitch=0.1,
                pace=1.2
            )
        )

        # For bulbul v3 beta with any speaker and temperature:
        # Note: pace and loudness are not supported for bulbul v3 and bulbul v3 beta
        tts_v3 = SarvamTTSService(
            api_key="your-api-key",
            voice_id="speaker_name",
            model="bulbul:v3",
            params=SarvamTTSService.InputParams(
                language=Language.HI,
                temperature=0.8
            )
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Sarvam TTS.

        Parameters:
            pitch: Voice pitch adjustment (-0.75 to 0.75). Defaults to 0.0.
            pace: Speech pace multiplier (0.3 to 3.0). Defaults to 1.0.
            loudness: Volume multiplier (0.1 to 3.0). Defaults to 1.0.
            enable_preprocessing: Enable text preprocessing. Defaults to False.
            min_buffer_size: Minimum number of characters to buffer before generating audio.
                Lower values reduce latency but may affect quality. Defaults to 50.
            max_chunk_length: Maximum number of characters processed in a single chunk.
                Controls memory usage and processing efficiency. Defaults to 200.
            output_audio_codec: Audio codec format. Defaults to "linear16".
            output_audio_bitrate: Audio bitrate. Defaults to "128k".
            language: Target language for synthesis. Supports Bengali (bn-IN), English (en-IN),
                Gujarati (gu-IN), Hindi (hi-IN), Kannada (kn-IN), Malayalam (ml-IN),
                Marathi (mr-IN), Odia (od-IN), Punjabi (pa-IN), Tamil (ta-IN),
                Telugu (te-IN). Defaults to en-IN.

                Available Speakers:
            Female: anushka, manisha, vidya, arya
            Male: abhilash, karun, hitesh
        """

        pitch: Optional[float] = Field(default=0.0, ge=-0.75, le=0.75)
        pace: Optional[float] = Field(default=1.0, ge=0.3, le=3.0)
        loudness: Optional[float] = Field(default=1.0, ge=0.1, le=3.0)
        enable_preprocessing: Optional[bool] = False
        min_buffer_size: Optional[int] = 50
        max_chunk_length: Optional[int] = 200
        output_audio_codec: Optional[str] = "linear16"
        output_audio_bitrate: Optional[str] = "128k"
        language: Optional[Language] = Language.EN
        temperature: Optional[float] = Field(
            default=0.6,
            ge=0.01,
            le=1.0,
            description="Controls the randomness of the output for bulbul v3 beta. "
            "Lower values make the output more focused and deterministic, while "
            "higher values make it more random. Range: 0.01 to 1.0. Default: 0.6.",
        )

    Settings = SarvamTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "bulbul:v2",
        voice_id: str = "anushka",
        url: str = "wss://api.sarvam.ai/text-to-speech/ws",
        aggregate_sentences: Optional[bool] = True,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Sarvam TTS service with voice and transport configuration.

        Args:
            api_key: Sarvam API key for authenticating TTS requests.
            model: Identifier of the Sarvam speech model (default "bulbul:v2").
                Supports "bulbul:v2", "bulbul:v3-beta" and "bulbul:v3".
            voice_id: Voice identifier for synthesis (default "anushka").
            url: WebSocket URL for connecting to the TTS backend (default production URL).
            aggregate_sentences: Whether to merge multiple sentences into one audio chunk (default True).
            sample_rate: Desired sample rate for the output audio in Hz (overrides default if set).
            params: Optional input parameters to override global configuration.
            settings: Runtime-updatable settings. When provided, values take precedence
                over ``voice_id`` and ``model``.
            **kwargs: Optional keyword arguments forwarded to InterruptibleTTSService (such as
                `push_stop_frames`, `sample_rate`, task manager parameters, event hooks, etc.)
                to customize transport behavior or enable metrics support.

        This method sets up the internal TTS configuration mapping, constructs the WebSocket
        URL based on the chosen model, and initializes state flags before connecting.
        """
        params = params or SarvamTTSService.InputParams()

        resolved_language = (
            language_to_sarvam_language(params.language) if params.language else "en-IN"
        )

        default_settings = self.Settings(
            model=model,
            voice=voice_id,
            language=resolved_language,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=True,
            pause_frame_processing=True,
            push_stop_frames=True,
            push_start_frame=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._websocket_url = f"{url}?model={default_settings.model}"
        self._api_key = api_key

        # Outgoing config message body — separate concern from framework Settings.
        self._request_payload: dict = {
            "target_language_code": default_settings.language,
            "speaker": default_settings.voice,
            "speech_sample_rate": 0,
            "enable_preprocessing": params.enable_preprocessing,
            "min_buffer_size": params.min_buffer_size,
            "max_chunk_length": params.max_chunk_length,
            "output_audio_codec": params.output_audio_codec,
            "output_audio_bitrate": params.output_audio_bitrate,
        }

        if default_settings.model in ("bulbul:v3-beta", "bulbul:v3"):
            self._request_payload.update(
                {
                    "temperature": getattr(params, "temperature", 0.6),
                    "model": default_settings.model,
                }
            )
        else:
            self._request_payload.update(
                {
                    "pitch": params.pitch,
                    "pace": params.pace,
                    "loudness": params.loudness,
                    "model": default_settings.model,
                }
            )

        self._receive_task = None
        self._keepalive_task = None
        self._disconnecting = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def start(self, frame: StartFrame):
        """Start the Sarvam TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        self._request_payload["speech_sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio synthesis by sending stop command."""
        if self._websocket:
            msg = {"type": "flush"}
            await self._websocket.send(json.dumps(msg))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and flush audio if it's the end of a full response."""
        await super().process_frame(frame, direction)

        # When the LLM finishes responding, flush any remaining text in Sarvam's buffer
        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def _update_settings(self, delta: TTSSettings) -> dict:
        """Apply a settings delta and reconnect if voice changed."""
        changed = await super()._update_settings(delta)
        if "voice" in changed:
            self._request_payload["speaker"] = self._settings.voice
            logger.info(f"Switching TTS voice to: [{self._settings.voice}]")
            await self._send_config()
        return changed

    async def _connect(self):
        """Connect to Sarvam WebSocket and start background tasks."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(
                self._keepalive_task_handler(),
            )

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket and clean up tasks."""
        await super()._disconnect()

        try:
            # First, set a flag to prevent new operations
            self._disconnecting = True

            # Cancel background tasks BEFORE closing websocket
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None

            if self._keepalive_task:
                await self.cancel_task(self._keepalive_task, timeout=2.0)
                self._keepalive_task = None

            # Now close the websocket
            await self._disconnect_websocket()

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            # Reset state only after everything is cleaned up
            self._websocket = None
            self._disconnecting = False

    async def _connect_websocket(self):
        """Establish WebSocket connection to Sarvam API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._websocket = await websocket_connect(
                self._websocket_url,
                additional_headers={
                    "api-subscription-key": self._api_key,
                    **sdk_headers(),
                },
            )
            logger.debug("Connected to Sarvam TTS Websocket")
            await self._send_config()

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to Sarvam TTS Websocket: {e}", exception=e
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _send_config(self):
        """Send initial configuration message."""
        if not self._websocket:
            raise Exception("WebSocket not connected")
        self._request_payload["speaker"] = self._settings.voice
        logger.debug(f"Config being sent is {self._request_payload}")
        config_message = {"type": "config", "data": self._request_payload}

        try:
            await self._websocket.send(json.dumps(config_message))
            logger.debug("Configuration sent successfully")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Sarvam")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from Sarvam WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, str):
                msg = json.loads(message)
                if msg.get("type") == "audio":
                    audio = base64.b64decode(msg["data"]["audio"])
                    ctx_id = self.get_active_audio_context_id()
                    await self.stop_ttfb_metrics(context_id=ctx_id)
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=ctx_id)
                    await self.append_to_audio_context(ctx_id, frame)
                elif msg.get("type") == "error":
                    error_msg = msg["data"]["message"]
                    await self.push_error(error_msg=f"TTS Error: {error_msg}")

                    # If it's a timeout error, the connection might need to be reset
                    if "too long" in error_msg.lower() or "timeout" in error_msg.lower():
                        logger.warning("Connection timeout detected, service may need restart")

                    await self.push_frame(ErrorFrame(error=f"TTS Error: {error_msg}"))

    async def _keepalive_task_handler(self):
        """Handle keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 20
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send keepalive message to maintain connection."""
        if self._disconnecting:
            return

        if self._websocket and self._websocket.state == State.OPEN:
            msg = {"type": "ping"}
            await self._websocket.send(json.dumps(msg))

    async def _send_text(self, text: str):
        """Send text to Sarvam WebSocket for synthesis."""
        if self._disconnecting:
            logger.warning("Service is disconnecting, ignoring text send")
            return

        if self._websocket and self._websocket.state == State.OPEN:
            msg = {"type": "text", "data": {"text": text}}
            await self._websocket.send(json.dumps(msg))
        else:
            logger.warning("WebSocket not ready, cannot send text")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech audio frames from input text using Sarvam TTS.

        Sends text over the WebSocket; audio frames are delivered via the receive task
        into the active audio context.

        Args:
            text: The text input to synthesize.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: ``None`` while audio is streamed asynchronously, or an ErrorFrame on failure.
        """
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self.start_ttfb_metrics(context_id=context_id)
                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
