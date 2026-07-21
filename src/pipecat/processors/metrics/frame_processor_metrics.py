#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processor metrics collection and reporting."""

import time
from typing import Dict

from loguru import logger

from pipecat.audio.utils import detect_speech_onset
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.utils.base_object import BaseObject


class FrameProcessorMetrics(BaseObject):
    """Metrics collection and reporting for frame processors.

    Provides comprehensive metrics tracking for frame processing operations,
    including timing measurements, resource usage, and performance analytics.
    Supports TTFB tracking, TTFA tracking, processing duration metrics, and
    usage statistics for LLM and TTS operations.

    TTFB and TTFA state is tracked per-context so overlapping TTS chunks
    (each running in its own asyncio task keyed by ``context_id``) can't
    overwrite each other's start times. Callers that don't pass
    ``context_id`` share a single sentinel key, matching pre-existing
    behavior for LLM/STT.
    """

    # Cap on buffered audio while waiting for a TTFA speech onset, so a response
    # that is all silence (or never becomes audible) can't grow the buffer without
    # bound.
    _TTFA_MAX_BUFFER_SECONDS = 3.0

    def __init__(self):
        """Initialize the frame processor metrics collector.

        Sets up internal state for tracking various metrics including TTFB,
        processing times, and usage statistics.
        """
        super().__init__()
        self._start_ttfb_times: Dict[str, float] = {}
        self._start_processing_time = 0
        self._start_text_aggregation_time = 0
        self._last_ttfb_time = 0
        self._should_report_ttfb = True
        # TTFA (time-to-first-audio) state, keyed by context_id. TTFA extends
        # TTFB: scanning for the first audible sample begins when TTFB stops
        # (see stop_ttfb_metrics). Audio is buffered across chunks until a
        # sustained onset is confirmed.
        self._ttfa_active: Dict[str, bool] = {}
        self._ttfa_buffer: Dict[str, bytes] = {}

    @property
    def ttfb(self) -> float | None:
        """Get the current TTFB value in seconds.

        Returns:
            The TTFB value in seconds, or None if not measured.
        """
        if self._last_ttfb_time > 0:
            return self._last_ttfb_time

        # If any TTFB measurement is in progress, use the earliest start
        # so the reported "in-progress" value is conservative under overlap.
        if self._start_ttfb_times:
            earliest = min(self._start_ttfb_times.values())
            if earliest > 0:
                return time.time() - earliest

        return None

    def _processor_name(self):
        """Get the processor name from core metrics data."""
        return self._core_metrics_data.processor

    def _model_name(self):
        """Get the model name from core metrics data."""
        return self._core_metrics_data.model

    @staticmethod
    def _ctx_key(context_id: str | None) -> str:
        """Map a context_id to the dict key used for per-context state.

        ``None`` collapses to a shared sentinel key so LLM/STT (which never
        pass a context_id) keep their pre-existing single-slot semantics.
        """
        return context_id or ""

    def _reset_ttfa(self, key: str):
        """Clear TTFA scan state for a given key."""
        self._ttfa_active.pop(key, None)
        self._ttfa_buffer.pop(key, None)

    def set_core_metrics_data(self, data: MetricsData):
        """Set the core metrics data for this collector.

        Args:
            data: The core metrics data containing processor and model information.
        """
        self._core_metrics_data = data

    def set_processor_name(self, name: str):
        """Set the processor name for metrics reporting.

        Args:
            name: The name of the processor to use in metrics.
        """
        self._core_metrics_data = MetricsData(processor=name)

    async def start_ttfb_metrics(
        self,
        *,
        start_time: float | None = None,
        report_only_initial_ttfb: bool,
        context_id: str | None = None,
    ):
        """Start measuring time-to-first-byte (TTFB).

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
            report_only_initial_ttfb: Whether to report only the first TTFB measurement.
            context_id: Optional per-context key. When set, this measurement
                is stored independently from other in-flight contexts so
                overlapping TTS chunks don't overwrite each other's start
                time. When ``None``, a shared sentinel key is used.
        """
        if self._should_report_ttfb:
            self._start_ttfb_times[self._ctx_key(context_id)] = start_time or time.time()
            self._last_ttfb_time = 0
            self._should_report_ttfb = not report_only_initial_ttfb

    async def stop_ttfb_metrics(
        self, *, end_time: float | None = None, context_id: str | None = None
    ):
        """Stop TTFB measurement and generate metrics frame.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
            context_id: Optional per-context key that must match the one
                passed to :meth:`start_ttfb_metrics`. When ``None``, the
                shared sentinel key is used.

        Returns:
            MetricsFrame containing TTFB data, or None if not measuring.
        """
        key = self._ctx_key(context_id)
        start = self._start_ttfb_times.pop(key, 0)
        if start == 0:
            return None

        end_time = end_time or time.time()

        self._last_ttfb_time = end_time - start
        logger.debug(f"{self._processor_name()} TTFB: {self._last_ttfb_time:.3f}s")
        ttfb = TTFBMetricsData(
            processor=self._processor_name(), value=self._last_ttfb_time, model=self._model_name()
        )
        # The first byte for this context has arrived; begin scanning leading
        # silence so TTFA can be reported as TTFB plus the silence duration
        # (see process_ttfa_metrics).
        self._ttfa_active[key] = True
        self._ttfa_buffer[key] = b""
        return MetricsFrame(data=[ttfb])

    async def process_ttfa_metrics(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        num_channels: int,
        context_id: str | None = None,
    ):
        """Scan buffered audio for speech onset and report TTFA.

        TTFA extends TTFB: once :meth:`stop_ttfb_metrics` records the first byte,
        audio is buffered across chunks (the leading silence may span several,
        and onset confirmation needs a little audio past it) and scanned for a
        sustained speech onset. When found, TTFB plus the leading-silence
        duration is emitted and measuring stops until the next response.

        Args:
            audio: Raw PCM audio bytes (16-bit signed integers).
            sample_rate: Sample rate of the audio in Hz.
            num_channels: Number of interleaved audio channels.
            context_id: Optional per-context key matching the one used for
                TTFB. When ``None``, the shared sentinel key is used.

        Returns:
            MetricsFrame containing TTFA data once onset is confirmed, otherwise
            None.
        """
        key = self._ctx_key(context_id)
        if not self._ttfa_active.get(key) or sample_rate <= 0:
            return None

        buffer = self._ttfa_buffer.get(key, b"") + audio
        self._ttfa_buffer[key] = buffer
        onset = detect_speech_onset(buffer, sample_rate, num_channels)
        if onset is None:
            # No confirmed onset yet. Bound memory against pathologically long
            # (or never-arriving) silence by giving up past a sane limit.
            max_bytes = int(self._TTFA_MAX_BUFFER_SECONDS * sample_rate * max(num_channels, 1) * 2)
            if len(buffer) >= max_bytes:
                logger.debug(
                    f"{self._processor_name()} TTFA: no onset within "
                    f"{self._TTFA_MAX_BUFFER_SECONDS:.0f}s of audio; not reporting"
                )
                self._reset_ttfa(key)
            return None

        silence_duration = onset / sample_rate
        value = self._last_ttfb_time + silence_duration
        logger.debug(
            f"{self._processor_name()} TTFA: {value:.3f}s ({silence_duration:.3f}s leading silence)"
        )
        ttfa = TTFAMetricsData(
            processor=self._processor_name(),
            ttfa=value,
            leading_silence=silence_duration,
            ttfb=self._last_ttfb_time,
            model=self._model_name(),
        )
        self._reset_ttfa(key)
        return MetricsFrame(data=[ttfa])

    def discard_context_metrics(self, context_id: str | None = None):
        """Drop any in-flight TTFB/TTFA state for a given context.

        Called when a TTS context is torn down (completion or interruption)
        so an orphaned start time doesn't leak into the next measurement.

        Args:
            context_id: Context key to discard. When ``None``, the shared
                sentinel key is used.
        """
        key = self._ctx_key(context_id)
        self._start_ttfb_times.pop(key, None)
        self._reset_ttfa(key)

    async def start_processing_metrics(self, *, start_time: float | None = None):
        """Start measuring processing time.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        self._start_processing_time = start_time or time.time()

    async def stop_processing_metrics(self, *, end_time: float | None = None):
        """Stop processing time measurement and generate metrics frame.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.

        Returns:
            MetricsFrame containing processing duration data, or None if not measuring.
        """
        if self._start_processing_time == 0:
            return None

        end_time = end_time or time.time()

        value = end_time - self._start_processing_time
        logger.debug(f"{self._processor_name()} processing time: {value:.3f}s")
        processing = ProcessingMetricsData(
            processor=self._processor_name(), value=value, model=self._model_name()
        )
        self._start_processing_time = 0
        return MetricsFrame(data=[processing])

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Record LLM token usage metrics.

        Args:
            tokens: Token usage information including prompt and completion tokens.

        Returns:
            MetricsFrame containing LLM usage data.
        """
        logstr = f"{self._processor_name()} prompt tokens: {tokens.prompt_tokens}, completion tokens: {tokens.completion_tokens}"
        if tokens.cache_read_input_tokens:
            logstr += f", cache read input tokens: {tokens.cache_read_input_tokens}"
        if tokens.reasoning_tokens:
            logstr += f", reasoning tokens: {tokens.reasoning_tokens}"
        if tokens.input_audio_tokens:
            logstr += f", input audio tokens: {tokens.input_audio_tokens}"
        if tokens.output_audio_tokens:
            logstr += f", output audio tokens: {tokens.output_audio_tokens}"
        if tokens.cache_read_input_audio_tokens:
            logstr += f", cache read input audio tokens: {tokens.cache_read_input_audio_tokens}"
        logger.debug(logstr)
        value = LLMUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=tokens
        )
        return MetricsFrame(data=[value])

    async def start_tts_usage_metrics(self, text: str):
        """Record TTS character usage metrics.

        Args:
            text: The text being processed by TTS.

        Returns:
            MetricsFrame containing TTS usage data.
        """
        characters = TTSUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=len(text)
        )
        logger.debug(f"{self._processor_name()} usage characters: {characters.value}")
        return MetricsFrame(data=[characters])

    async def start_text_aggregation_metrics(self):
        """Start measuring text aggregation time (first token to first sentence)."""
        self._start_text_aggregation_time = time.time()

    async def stop_text_aggregation_metrics(self):
        """Stop text aggregation measurement and generate metrics frame.

        Returns:
            MetricsFrame containing text aggregation time, or None if not measuring.
        """
        if self._start_text_aggregation_time == 0:
            return None

        value = time.time() - self._start_text_aggregation_time
        logger.debug(f"{self._processor_name()} text aggregation time: {value}")
        aggregation = TextAggregationMetricsData(
            processor=self._processor_name(), value=value, model=self._model_name()
        )
        self._start_text_aggregation_time = 0
        return MetricsFrame(data=[aggregation])
