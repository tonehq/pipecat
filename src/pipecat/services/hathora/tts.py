#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""[Hathora-hosted](https://models.hathora.dev) text-to-speech services."""

import io
import os
import wave
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Tuple

import aiohttp
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from .utils import ConfigOption


def _decode_audio_payload(
    audio_bytes: bytes,
    *,
    fallback_sample_rate: int = 24000,
    fallback_channels: int = 1,
) -> Tuple[bytes, int, int]:
    """Convert a WAV/PCM payload into raw PCM samples for TTSAudioRawFrame."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_reader:
            channels = wav_reader.getnchannels()
            sample_rate = wav_reader.getframerate()
            frames = wav_reader.readframes(wav_reader.getnframes())
            return frames, sample_rate, channels
    except (wave.Error, EOFError):
        # If the payload is already raw PCM, just pass it through.
        return audio_bytes, fallback_sample_rate, fallback_channels


@dataclass
class HathoraTTSSettings(TTSSettings):
    """Settings for HathoraTTSService."""

    pass


class HathoraTTSService(TTSService):
    """This service supports several different text-to-speech models hosted by Hathora.

    [Documentation](https://models.hathora.dev)
    """

    Settings = HathoraTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Optional input parameters for Hathora TTS configuration.

        Parameters:
            speed: Speech speed multiplier (if supported by model).
            config: Some models support additional config, refer to
                [docs](https://models.hathora.dev) for each model to see
                what is supported.
        """

        speed: Optional[float] = None
        config: Optional[list[ConfigOption]] = None

    def __init__(
        self,
        *,
        model: str,
        voice_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.models.hathora.dev/inference/v1/tts",
        params: Optional[InputParams] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Hathora TTS service.

        Args:
            model: Model to use; find available models
                [here](https://models.hathora.dev).
            voice_id: Voice to use for synthesis (if supported by model).
            sample_rate: Output sample rate for generated audio.
            api_key: API key for authentication with the Hathora service;
                provision one [here](https://models.hathora.dev/tokens).
            base_url: Base API URL for the Hathora TTS service.
            params: Configuration parameters.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent class.
        """
        params = params or HathoraTTSService.InputParams()

        default_settings = self.Settings(
            model=model,
            voice=voice_id,
            language=None,
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

        self._api_key = api_key or os.getenv("HATHORA_API_KEY")
        self._base_url = base_url

        # Outgoing API request extras — separate concern from framework Settings.
        self._request_extras: dict = {
            "speed": params.speed,
            "config": params.config,
        }

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True
        """
        return True

    @classmethod
    def get_voices(self, api_key: str):
        return []

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Run text-to-speech synthesis on the provided text.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics(context_id=context_id)

            url = f"{self._base_url}"

            payload = {"model": self._settings.model, "text": text}

            if self._settings.voice is not None:
                payload["voice"] = self._settings.voice
            if self._request_extras["speed"] is not None:
                payload["speed"] = self._request_extras["speed"]
            if self._request_extras["config"] is not None:
                payload["model_config"] = [
                    {"name": option.name, "value": option.value}
                    for option in self._request_extras["config"]
                ]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept-Encoding": "gzip, deflate",
                    },
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"Hathora TTS API error {resp.status}: {error_text}"
                        )
                    audio_data = await resp.read()

            pcm_audio, sample_rate, num_channels = _decode_audio_payload(
                audio_data,
                fallback_sample_rate=self.sample_rate,
            )

            await self.stop_ttfb_metrics(context_id=context_id)

            yield TTSAudioRawFrame(
                audio=pcm_audio,
                sample_rate=self.sample_rate,
                num_channels=num_channels,
                context_id=context_id,
            )

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_processing_metrics()
