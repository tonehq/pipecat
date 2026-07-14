#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio text-to-speech service implementation.

This module provides integration with Fish Audio's real-time TTS WebSocket API
for streaming text-to-speech synthesis with customizable voice parameters.
"""

from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import ormsgpack
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Fish Audio, you need to `uv add "pipecat-ai[fish]"`.')
    raise ImportError(f"Missing module: {e}") from e

# FishAudio supports various output formats
FishAudioOutputFormat = Literal["opus", "mp3", "pcm", "wav"]


@dataclass
class FishAudioTTSSettings(TTSSettings):
    """Settings for FishAudioTTSService.

    Parameters:
        latency: Latency mode ("normal" or "balanced"). Defaults to "balanced".
        normalize: Whether to normalize audio output. Defaults to True.
        temperature: Controls randomness in speech generation (0.0-1.0).
        top_p: Controls diversity via nucleus sampling (0.0-1.0).
        prosody_speed: Speech speed multiplier (0.5-2.0). Defaults to 1.0.
        prosody_volume: Volume adjustment in dB (-20 to 20). Defaults to 0.
    """

    latency: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    normalize: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_volume: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> Self:
        """Construct settings from a plain dict, destructuring legacy nested ``prosody``."""
        flat = dict(settings)
        nested = flat.pop("prosody", None)
        if isinstance(nested, dict):
            flat.setdefault("prosody_speed", nested.get("speed"))
            flat.setdefault("prosody_volume", nested.get("volume"))
        return super().from_mapping(flat)


class FishAudioTTSService(InterruptibleTTSService):
    """Fish Audio text-to-speech service with WebSocket streaming.

    Provides real-time text-to-speech synthesis using Fish Audio's WebSocket API.
    Supports various audio formats, customizable prosody controls, and streaming
    audio generation with interruption handling.
    """

    Settings = FishAudioTTSSettings
    _settings: Settings

    @deprecated(
        "`FishAudioTTSService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `FishAudioTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for Fish Audio TTS configuration.

        .. deprecated:: 0.0.105
            Use ``settings=FishAudioTTSService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language for synthesis. Defaults to English.
            latency: Latency mode ("normal" or "balanced"). Defaults to "normal".
            normalize: Whether to normalize audio output. Defaults to True.
            prosody_speed: Speech speed multiplier (0.5-2.0). Defaults to 1.0.
            prosody_volume: Volume adjustment in dB. Defaults to 0.
        """

        language: Language | None = Language.EN
        latency: str | None = "normal"  # "normal" or "balanced"
        normalize: bool | None = True
        prosody_speed: float | None = 1.0  # Speech speed (0.5-2.0)
        prosody_volume: int | None = 0  # Volume adjustment in dB

    def __init__(
        self,
        *,
        api_key: str,
        reference_id: str | None = None,  # This is the voice ID
        model_id: str | None = None,
        output_format: FishAudioOutputFormat = "pcm",
        sample_rate: int | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Fish Audio TTS service.

        Args:
            api_key: Fish Audio API key for authentication.
            reference_id: Reference ID of the voice model to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            model_id: Specify which Fish Audio TTS model to use (e.g. "s1").

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            output_format: Audio output format. Defaults to "pcm".
            sample_rate: Audio sample rate. If None, uses default.
            params: Additional input parameters for voice customization.

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent service.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="s2-pro",
            voice=None,
            language=None,
            latency="balanced",
            normalize=True,
            temperature=None,
            top_p=None,
            prosody_speed=1.0,
            prosody_volume=0,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if reference_id is not None:
            self._warn_init_param_moved_to_settings("reference_id", "voice")
            default_settings.voice = reference_id
        if model_id is not None:
            self._warn_init_param_moved_to_settings("model_id", "model")
            default_settings.model = model_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.latency is not None:
                    default_settings.latency = params.latency
                if params.normalize is not None:
                    default_settings.normalize = params.normalize
                if params.prosody_speed is not None:
                    default_settings.prosody_speed = params.prosody_speed
                if params.prosody_volume is not None:
                    default_settings.prosody_volume = params.prosody_volume

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = "wss://api.fish.audio/v1/tts/live"
        self._websocket = None
        self._receive_task = None

        # Init-only audio format config (not runtime-updatable).
        self._fish_sample_rate = 0  # Set in start()
        self._output_format = output_format

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Fish Audio service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if needed.

        Any change to voice or model triggers a WebSocket reconnect.

        Args:
            delta: A :class:`TTSSettings` (or ``FishAudioTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def start(self, frame: StartFrame):
        """Start the Fish Audio TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._fish_sample_rate = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Fish Audio TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Fish Audio TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Fish Audio")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            model = assert_given(self._settings.model)
            if model is not None:
                headers["model"] = model
            websocket = await websocket_connect(self._base_url, additional_headers=headers)
            self._websocket = websocket

            # Send initial start message with ormsgpack
            request_settings = {
                "sample_rate": self._fish_sample_rate,
                "latency": self._settings.latency,
                "format": self._output_format,
                "normalize": self._settings.normalize,
                "prosody": {
                    "speed": self._settings.prosody_speed,
                    "volume": self._settings.prosody_volume,
                },
                "reference_id": self._settings.voice,
            }
            if self._settings.temperature is not None:
                request_settings["temperature"] = self._settings.temperature
            if self._settings.top_p is not None:
                request_settings["top_p"] = self._settings.top_p
            start_message = {"event": "start", "request": {"text": "", **request_settings}}
            await websocket.send(ormsgpack.packb(start_message))
            logger.debug("Sent start message to Fish Audio")

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Fish Audio")
                # Send stop event with ormsgpack
                stop_message = {"event": "stop"}
                await self._websocket.send(ormsgpack.packb(stop_message))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def flush_audio(self, context_id: str | None = None):
        """Flush any buffered audio by sending a flush event to Fish Audio."""
        logger.trace(f"{self}: Flushing audio buffers")
        if not self._websocket or self._websocket.state is State.CLOSED:
            return
        flush_message = {"event": "flush"}
        await self._get_websocket().send(ormsgpack.packb(flush_message))

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        """Stop all metrics when audio context is interrupted."""
        await self.stop_all_metrics()
        await super().on_audio_context_interrupted(context_id)

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                if isinstance(message, bytes):
                    msg = ormsgpack.unpackb(message)
                    if isinstance(msg, dict):
                        event = msg.get("event")
                        if event == "audio":
                            audio_data = msg.get("audio")
                            # Only process larger chunks to remove msgpack overhead
                            if audio_data and len(audio_data) > 1024:
                                context_id = self.get_active_audio_context_id()
                                frame = TTSAudioRawFrame(
                                    audio_data,
                                    self.sample_rate,
                                    1,
                                    context_id=context_id,
                                )
                                await self.append_to_audio_context(context_id, frame)
                                await self.stop_ttfb_metrics(context_id=context_id)
                        elif event == "finish":
                            reason = msg.get("reason", "unknown")
                            if reason == "error":
                                await self.push_error(
                                    error_msg="Fish Audio server error during synthesis"
                                )
                            else:
                                logger.debug(f"Fish Audio session finished: {reason}")

            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Fish Audio's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames and control frames for the synthesized speech.
        """
        logger.debug(f"{self}: Generating Fish TTS: [{text}]")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            # Send the text
            text_message = {
                "event": "text",
                "text": text,
            }
            try:
                await self._get_websocket().send(ormsgpack.packb(text_message))
                await self.start_tts_usage_metrics(text)

                # Send flush event to force audio generation
                flush_message = {"event": "flush"}
                await self._get_websocket().send(ormsgpack.packb(flush_message))
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()

            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    @classmethod
    def get_voices(cls, api_key: str):
        return [
            {"name": "Friendly Person", "voice_id": "54a5170264694bfc8e9ad98df7bd89c3", "description": "A friendly and engaging English speaker", "gender": "male", "language": "en", "sample_url": None, "accent": None},
            {"name": "Anime Girl", "voice_id": "7f92f8efb8ec43bf8f646fd6d4a6af54", "description": "Anime-style female English voice", "gender": "female", "language": "en", "sample_url": None, "accent": None},
            {"name": "Nature Documentary", "voice_id": "0eb2bd3576714dbcad7cd4c6b2b6e12f", "description": "Deep, authoritative narrator voice", "gender": "male", "language": "en", "sample_url": None, "accent": None},
            {"name": "Child", "voice_id": "5c7b3dba4e0a4f03b92a7e0e7b498a6f", "description": "Young child voice for English", "gender": "male", "language": "en", "sample_url": None, "accent": None},
            {"name": "Cute Girl", "voice_id": "c5e2e78f6b4946a0b28d6f9ee0a6f4ad", "description": "Cute, expressive female English voice", "gender": "female", "language": "en", "sample_url": None, "accent": None},
            {"name": "Calm Female", "voice_id": "ad3b294dacf14e8e9fd0924fcc7d79c2", "description": "Calm, soothing female English voice", "gender": "female", "language": "en", "sample_url": None, "accent": None},
            {"name": "News Anchor", "voice_id": "66e038ab7ef84e27a4d96b1b8f49eb4d", "description": "Professional news anchor voice", "gender": "male", "language": "en", "sample_url": None, "accent": None},
            {"name": "Chinese Female", "voice_id": "a1d6b0d67e2c4f78a9c0b1e3d5f7c812", "description": "Standard Mandarin female voice", "gender": "female", "language": "zh", "sample_url": None, "accent": None},
            {"name": "Chinese Male", "voice_id": "b3e7c2a14d8f4e92b5c6d7a8f9e0b1c3", "description": "Standard Mandarin male voice", "gender": "male", "language": "zh", "sample_url": None, "accent": None},
            {"name": "Japanese Female", "voice_id": "d4f8a1c2e3b5f6d7a8c9e0b1d2f3a4c5", "description": "Standard Japanese female voice", "gender": "female", "language": "ja", "sample_url": None, "accent": None},
        ]
