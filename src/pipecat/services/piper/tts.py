#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Piper TTS service implementation."""

from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


# This assumes a running TTS service running: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
class PiperTTSService(TTSService):
    """Piper TTS service implementation.

    Provides integration with Piper's HTTP TTS server for text-to-speech
    synthesis. Supports streaming audio generation with configurable sample
    rates and automatic WAV header removal.
    """

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        # When using Piper, the sample rate of the generated audio depends on the
        # voice model being used.
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the Piper TTS service.

        Args:
            base_url: Base URL for the Piper TTS HTTP server.
            aiohttp_session: aiohttp ClientSession for making HTTP requests.
            sample_rate: Output sample rate. If None, uses the voice model's native rate.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        if base_url.endswith("/"):
            logger.warning("Base URL ends with a slash, this is not allowed.")
            base_url = base_url[:-1]

        self._base_url = base_url
        self._session = aiohttp_session
        self._settings = {"base_url": base_url}

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Piper service supports metrics generation.
        """
        return True

    @classmethod
    def get_voices(self, api_key: str):
        return []

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Piper's HTTP API.

        Args:
            text: The text to convert to speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        headers = {
            "Content-Type": "application/json",
        }
        try:
            await self.start_ttfb_metrics(context_id=context_id)

            async with self._session.post(
                self._base_url, json={"text": text}, headers=headers
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                CHUNK_SIZE = self.chunk_size

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(CHUNK_SIZE), strip_wav_header=True
                ):
                    await self.stop_ttfb_metrics(context_id=context_id)
                    yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            # No explicit stop_ttfb_metrics here: the base class closes TTFB when
            # it forwards the first TTSAudioRawFrame in _handle_audio_context.
            # Stopping in a finally block also fired on error paths where no audio
            # was produced, emitting a value that measured the failure latency
            # rather than a real time-to-first-byte.
            logger.debug(f"{self}: Finished TTS [{text}]")
            yield TTSStoppedFrame()
