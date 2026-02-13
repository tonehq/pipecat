#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Riva text-to-speech service implementation.

This module provides integration with NVIDIA Riva's TTS services through
gRPC API for high-quality speech synthesis.
"""

import asyncio
import os
from typing import AsyncGenerator, AsyncIterable, Generator, Mapping, Optional

from pipecat.utils.tracing.service_decorators import traced_tts

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

try:
    import riva.client
    import riva.client.proto.riva_tts_pb2 as rtts
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use NVIDIA Riva TTS, you need to `pip install pipecat-ai[nvidia]`.")
    raise Exception(f"Missing module: {e}")


class NvidiaTTSService(TTSService):
    """NVIDIA Riva text-to-speech service.

    Provides high-quality text-to-speech synthesis using NVIDIA Riva's
    cloud-based TTS models. Supports multiple voices, languages, and
    configurable quality settings.
    """

    class InputParams(BaseModel):
        """Input parameters for Riva TTS configuration.

        Parameters:
            language: Language code for synthesis. Defaults to US English.
            quality: Audio quality setting (0-100). Defaults to 20.
        """

        language: Optional[Language] = Language.EN_US
        quality: Optional[int] = 20

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "Magpie-Multilingual.EN-US.Aria",
        sample_rate: Optional[int] = None,
        model_function_map: Mapping[str, str] = {
            "function_id": "877104f7-e885-42b9-8de8-f6e4c6303969",
            "model_name": "magpie-tts-multilingual",
        },
        params: Optional[InputParams] = None,
        use_ssl: bool = True,
        **kwargs,
    ):
        """Initialize the NVIDIA Riva TTS service.

        Args:
            api_key: NVIDIA API key for authentication.
            server: gRPC server endpoint. Defaults to NVIDIA's cloud endpoint.
            voice_id: Voice model identifier. Defaults to multilingual Ray voice.
            sample_rate: Audio sample rate. If None, uses service default.
            model_function_map: Dictionary containing function_id and model_name for the TTS model.
            params: Additional configuration parameters for TTS synthesis.
            use_ssl: Whether to use SSL for the NVIDIA Riva server. Defaults to True.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or NvidiaTTSService.InputParams()

        self._server = server
        self._api_key = api_key
        self._voice_id = voice_id
        self._language_code = params.language
        self._quality = params.quality
        self._function_id = model_function_map.get("function_id")
        self._use_ssl = use_ssl
        self.set_model_name(model_function_map.get("model_name"))
        self.set_voice(voice_id)

        self._service = None
        self._config = None

    @classmethod
    def get_voices(cls, api_key: str):
        raw = [
  {
    "voice_id": "Magpie-Multilingual.EN-US.Female.Neutral",
    "name": "Magpie Multilingual EN-US Female Neutral",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Female Neutral emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Female.Calm",
    "name": "Magpie Multilingual EN-US Female Calm",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Female Calm emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Female.Fearful",
    "name": "Magpie Multilingual EN-US Female Fearful",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Female Fearful emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Female.Happy",
    "name": "Magpie Multilingual EN-US Female Happy",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Female Happy emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Female.Angry",
    "name": "Magpie Multilingual EN-US Female Angry",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Female Angry emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Female.Female-1",
    "name": "Magpie Multilingual EN-US Female 1",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Female voice 1",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/ad8bb6b5199a7579a34395e4f4a07ff0/EN-US.Female.Female-1_MagpieTTS.wav"
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Male.Calm",
    "name": "Magpie Multilingual EN-US Male Calm",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Male Calm emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Male.Fearful",
    "name": "Magpie Multilingual EN-US Male Fearful",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Male Fearful emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Male.Happy",
    "name": "Magpie Multilingual EN-US Male Happy",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Male Happy emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Male.Neutral",
    "name": "Magpie Multilingual EN-US Male Neutral",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Male Neutral emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Male.Angry",
    "name": "Magpie Multilingual EN-US Male Angry",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Male Angry emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Male.Disgusted",
    "name": "Magpie Multilingual EN-US Male Disgusted",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Male Disgusted emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.EN-US.Male.Male-1",
    "name": "Magpie Multilingual EN-US Male 1",
    "language": "en-US",
    "description": "Multilingual TTS model - English US Male voice 1",
    "accent": "US",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/7776fdc8bfb84bb6c091fab0ca6bc38d/EN-US.Male.Male-1_MagpieTTS.wav"
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Male.Male-1",
    "name": "Magpie Multilingual FR-FR Male 1",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Male voice 1",
    "accent": "FR",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/a6815f1502ff52c348487d07041f512d/FR-FR.Male.Male-1_MagpieTTS.wav"
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Female.Female-1",
    "name": "Magpie Multilingual FR-FR Female 1",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Female voice 1",
    "accent": "FR",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/54aa7b237b4b50c1e6cf83407fb5fdb7/FR-FR.Female.Female-1_MagpieTTS.wav"
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Female.Angry",
    "name": "Magpie Multilingual FR-FR Female Angry",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Female Angry emotion",
    "accent": "FR",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Female.Calm",
    "name": "Magpie Multilingual FR-FR Female Calm",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Female Calm emotion",
    "accent": "FR",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Female.Disgust",
    "name": "Magpie Multilingual FR-FR Female Disgust",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Female Disgust emotion",
    "accent": "FR",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Female.Sad",
    "name": "Magpie Multilingual FR-FR Female Sad",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Female Sad emotion",
    "accent": "FR",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Female.Happy",
    "name": "Magpie Multilingual FR-FR Female Happy",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Female Happy emotion",
    "accent": "FR",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Female.Fearful",
    "name": "Magpie Multilingual FR-FR Female Fearful",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Female Fearful emotion",
    "accent": "FR",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Female.Neutral",
    "name": "Magpie Multilingual FR-FR Female Neutral",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Female Neutral emotion",
    "accent": "FR",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Male.Neutral",
    "name": "Magpie Multilingual FR-FR Male Neutral",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Male Neutral emotion",
    "accent": "FR",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Male.Angry",
    "name": "Magpie Multilingual FR-FR Male Angry",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Male Angry emotion",
    "accent": "FR",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Male.Calm",
    "name": "Magpie Multilingual FR-FR Male Calm",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Male Calm emotion",
    "accent": "FR",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.FR-FR.Male.Sad",
    "name": "Magpie Multilingual FR-FR Male Sad",
    "language": "fr-FR",
    "description": "Multilingual TTS model - French France Male Sad emotion",
    "accent": "FR",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Male.Male-1",
    "name": "Magpie Multilingual ES-US Male 1",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Male voice 1",
    "accent": "US",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/45c1034ce0262627f2ed865f02db079e/ES-US.Male.Male-1_MagpieTTS.wav"
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Female.Female-1",
    "name": "Magpie Multilingual ES-US Female 1",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Female voice 1",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/04f92cc1cb8517c9ea1a959a8cd6ab3f/ES-US.Female.Female-1_MagpieTTS.wav"
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Female.Neutral",
    "name": "Magpie Multilingual ES-US Female Neutral",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Female Neutral emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Male.Neutral",
    "name": "Magpie Multilingual ES-US Male Neutral",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Male Neutral emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Male.Angry",
    "name": "Magpie Multilingual ES-US Male Angry",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Male Angry emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Female.Angry",
    "name": "Magpie Multilingual ES-US Female Angry",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Female Angry emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Female.Happy",
    "name": "Magpie Multilingual ES-US Female Happy",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Female Happy emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Male.Happy",
    "name": "Magpie Multilingual ES-US Male Happy",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Male Happy emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Female.Calm",
    "name": "Magpie Multilingual ES-US Female Calm",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Female Calm emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Male.Calm",
    "name": "Magpie Multilingual ES-US Male Calm",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Male Calm emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Female.Pleasant_Surprise",
    "name": "Magpie Multilingual ES-US Female Pleasant Surprise",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Female Pleasant Surprise emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Male.Pleasant_Surprise",
    "name": "Magpie Multilingual ES-US Male Pleasant Surprise",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Male Pleasant Surprise emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Female.Sad",
    "name": "Magpie Multilingual ES-US Female Sad",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Female Sad emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Male.Sad",
    "name": "Magpie Multilingual ES-US Male Sad",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Male Sad emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Magpie-Multilingual.ES-US.Male.Disgust",
    "name": "Magpie Multilingual ES-US Male Disgust",
    "language": "es-US",
    "description": "Multilingual TTS model - Spanish US Male Disgust emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "English-US.Female-1",
    "name": "English US Female 1",
    "language": "en-US",
    "description": "FastPitch model - English US Female voice 1",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/11ea8df05a1db82a1bc858d169cd0ff5/Female1.wav"
  },
  {
    "voice_id": "English-US.Male-1",
    "name": "English US Male 1",
    "language": "en-US",
    "description": "FastPitch model - English US Male voice 1",
    "accent": "US",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/dcd2538b961a0c4a0d9ec4ced1936030/Male1.wav"
  },
  {
    "voice_id": "English-US.Female-Calm",
    "name": "English US Female Calm",
    "language": "en-US",
    "description": "FastPitch model - English US Female Calm emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/0372495b794c463964bae5d26876c117/Female_Calm.wav"
  },
  {
    "voice_id": "English-US.Female-Neutral",
    "name": "English US Female Neutral",
    "language": "en-US",
    "description": "FastPitch model - English US Female Neutral emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/1a55a0af0d7017cb7cced0c2033a7829/Female_Neutral.wav"
  },
  {
    "voice_id": "English-US.Female-Happy",
    "name": "English US Female Happy",
    "language": "en-US",
    "description": "FastPitch model - English US Female Happy emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/96c95b4b83d732c7215ece5032a81909/Female_Happy.wav"
  },
  {
    "voice_id": "English-US.Female-Angry",
    "name": "English US Female Angry",
    "language": "en-US",
    "description": "FastPitch model - English US Female Angry emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/c70e42129891b79ca925b754a45a602c/Female_Angry.wav"
  },
  {
    "voice_id": "English-US.Female-Fearful",
    "name": "English US Female Fearful",
    "language": "en-US",
    "description": "FastPitch model - English US Female Fearful emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/f02d799b21fb8feec1423a166c8eea1b/Female_Fearful.wav"
  },
  {
    "voice_id": "English-US.Female-Sad",
    "name": "English US Female Sad",
    "language": "en-US",
    "description": "FastPitch model - English US Female Sad emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/108a7e60f4462125222e4e50d83616f3/Female_Sad.wav"
  },
  {
    "voice_id": "English-US.Male-Calm",
    "name": "English US Male Calm",
    "language": "en-US",
    "description": "FastPitch model - English US Male Calm emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/9eb29ec1e7a923a52dd0ad6d0f63778a/Male_Calm.wav"
  },
  {
    "voice_id": "English-US.Male-Neutral",
    "name": "English US Male Neutral",
    "language": "en-US",
    "description": "FastPitch model - English US Male Neutral emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/daa5dda58c98485bee550cc182a62613/Male_Neutral.wav"
  },
  {
    "voice_id": "English-US.Male-Happy",
    "name": "English US Male Happy",
    "language": "en-US",
    "description": "FastPitch model - English US Male Happy emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/aa934d33a11eb1531aa2612c08a63453/Male_Happy.wav"
  },
  {
    "voice_id": "English-US.Male-Angry",
    "name": "English US Male Angry",
    "language": "en-US",
    "description": "FastPitch model - English US Male Angry emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/62ba38f5c611a75c4e63818ce8284b4b/Male_Angry.wav"
  },
  {
    "voice_id": "English-US-RadTTS.Female-1",
    "name": "English US RadTTS Female 1",
    "language": "en-US",
    "description": "RadTTS model - English US Female voice 1",
    "accent": "US",
    "gender": "Female",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/f6be975c5ba8bf2d1b77fb320c56e43b/Female1_RadTTS.wav"
  },
  {
    "voice_id": "English-US-RadTTS.Male-1",
    "name": "English US RadTTS Male 1",
    "language": "en-US",
    "description": "RadTTS model - English US Male voice 1",
    "accent": "US",
    "gender": "Male",
    "preview_url": "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/_downloads/1f9a0e259c11d076063c2699b1b0430b/Male1_RadTTS.wav"
  },
  {
    "voice_id": "English-US-RadTTS.Female-Calm",
    "name": "English US RadTTS Female Calm",
    "language": "en-US",
    "description": "RadTTS model - English US Female Calm emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Female-Neutral",
    "name": "English US RadTTS Female Neutral",
    "language": "en-US",
    "description": "RadTTS model - English US Female Neutral emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Female-Happy",
    "name": "English US RadTTS Female Happy",
    "language": "en-US",
    "description": "RadTTS model - English US Female Happy emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Female-Angry",
    "name": "English US RadTTS Female Angry",
    "language": "en-US",
    "description": "RadTTS model - English US Female Angry emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Female-Fearful",
    "name": "English US RadTTS Female Fearful",
    "language": "en-US",
    "description": "RadTTS model - English US Female Fearful emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Female-Sad",
    "name": "English US RadTTS Female Sad",
    "language": "en-US",
    "description": "RadTTS model - English US Female Sad emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Male-Calm",
    "name": "English US RadTTS Male Calm",
    "language": "en-US",
    "description": "RadTTS model - English US Male Calm emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Male-Neutral",
    "name": "English US RadTTS Male Neutral",
    "language": "en-US",
    "description": "RadTTS model - English US Male Neutral emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Male-Happy",
    "name": "English US RadTTS Male Happy",
    "language": "en-US",
    "description": "RadTTS model - English US Male Happy emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "English-US-RadTTS.Male-Angry",
    "name": "English US RadTTS Male Angry",
    "language": "en-US",
    "description": "RadTTS model - English US Male Angry emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "ljspeech",
    "name": "LJSpeech",
    "language": "en-US",
    "description": "FastPitch model trained on LJSpeech dataset",
    "accent": "US",
    "gender": "",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Female-1",
    "name": "Mandarin CN Female 1",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Female voice 1",
    "accent": "CN",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Male-1",
    "name": "Mandarin CN Male 1",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Male voice 1",
    "accent": "CN",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Female-Calm",
    "name": "Mandarin CN Female Calm",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Female Calm emotion",
    "accent": "CN",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Female-Neutral",
    "name": "Mandarin CN Female Neutral",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Female Neutral emotion",
    "accent": "CN",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Male-Happy",
    "name": "Mandarin CN Male Happy",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Male Happy emotion",
    "accent": "CN",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Male-Fearful",
    "name": "Mandarin CN Male Fearful",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Male Fearful emotion",
    "accent": "CN",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Male-Sad",
    "name": "Mandarin CN Male Sad",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Male Sad emotion",
    "accent": "CN",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Male-Calm",
    "name": "Mandarin CN Male Calm",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Male Calm emotion",
    "accent": "CN",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Male-Neutral",
    "name": "Mandarin CN Male Neutral",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Male Neutral emotion",
    "accent": "CN",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Mandarin-CN.Male-Angry",
    "name": "Mandarin CN Male Angry",
    "language": "zh-CN",
    "description": "FastPitch model - Mandarin Chinese Male Angry emotion",
    "accent": "CN",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-ES-Female-1",
    "name": "Spanish ES Female 1",
    "language": "es-ES",
    "description": "FastPitch model - Spanish Spain Female voice 1",
    "accent": "ES",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-ES-Male-1",
    "name": "Spanish ES Male 1",
    "language": "es-ES",
    "description": "FastPitch model - Spanish Spain Male voice 1",
    "accent": "ES",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Female-1",
    "name": "Spanish US Female 1",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Female voice 1",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Male-1",
    "name": "Spanish US Male 1",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Male voice 1",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Female-Calm",
    "name": "Spanish US Female Calm",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Female Calm emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Male-Calm",
    "name": "Spanish US Male Calm",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Male Calm emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Female-Angry",
    "name": "Spanish US Female Angry",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Female Angry emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Male-Angry",
    "name": "Spanish US Male Angry",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Male Angry emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Female-Neutral",
    "name": "Spanish US Female Neutral",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Female Neutral emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Male-Neutral",
    "name": "Spanish US Male Neutral",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Male Neutral emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Female-Sad",
    "name": "Spanish US Female Sad",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Female Sad emotion",
    "accent": "US",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Male-Happy",
    "name": "Spanish US Male Happy",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Male Happy emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Male-Fearful",
    "name": "Spanish US Male Fearful",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Male Fearful emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Spanish-US.Male-Sad",
    "name": "Spanish US Male Sad",
    "language": "es-US",
    "description": "FastPitch model - Spanish US Male Sad emotion",
    "accent": "US",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "Italian-IT-Female-1",
    "name": "Italian IT Female 1",
    "language": "it-IT",
    "description": "FastPitch model - Italian Italy Female voice 1",
    "accent": "IT",
    "gender": "Female",
    "preview_url": ""
  },
  {
    "voice_id": "Italian-IT-Male-1",
    "name": "Italian IT Male 1",
    "language": "it-IT",
    "description": "FastPitch model - Italian Italy Male voice 1",
    "accent": "IT",
    "gender": "Male",
    "preview_url": ""
  },
  {
    "voice_id": "German-DE-Male-1",
    "name": "German DE Male 1",
    "language": "de-DE",
    "description": "FastPitch model - German Germany Male voice 1",
    "accent": "DE",
    "gender": "Male",
    "preview_url": ""
  }
]
        return [
            {
                "name": v["name"],
                "voice_id": v["voice_id"],
                "description": v.get("description") or None,
                "gender": v.get("gender"),
                "language": v.get("language"),
                "sample_url": v.get("preview_url") or None,
                "accent": v.get("accent") or None,
            }
            for v in raw
        ]

    async def set_model(self, model: str):
        """Attempt to set the TTS model.

        Note: Model cannot be changed after initialization for Riva service.

        Args:
            model: The model name to set (operation not supported).
        """
        logger.warning(f"Cannot set model after initialization. Set model and function id like so:")
        example = {"function_id": "<UUID>", "model_name": "<model_name>"}
        logger.warning(
            f"{self.__class__.__name__}(api_key=<api_key>, model_function_map={example})"
        )

    def _initialize_client(self):
        if self._service is not None:
            return

        metadata = [
            ["function-id", self._function_id],
            ["authorization", f"Bearer {self._api_key}"],
        ]
        auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)

        self._service = riva.client.SpeechSynthesisService(auth)

    def _create_synthesis_config(self):
        if not self._service:
            return

        # warm up the service
        config = self._service.stub.GetRivaSynthesisConfig(
            riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
        )
        return config

    async def start(self, frame: StartFrame):
        """Start the Cartesia TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._initialize_client()
        self._config = self._create_synthesis_config()
        logger.debug(f"Initialized NvidiaTTSService with model: {self.model_name}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using NVIDIA Riva TTS.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """

        def read_audio_responses() -> Generator[rtts.SynthesizeSpeechResponse, None, None]:
            responses = self._service.synthesize_online(
                text,
                self._voice_id,
                self._language_code,
                sample_rate_hz=self.sample_rate,
                zero_shot_audio_prompt_file=None,
                zero_shot_quality=self._quality,
                custom_dictionary={},
            )
            return responses

        def async_next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        async def async_iterator(iterator) -> AsyncIterable[rtts.SynthesizeSpeechResponse]:
            while True:
                item = await asyncio.to_thread(async_next, iterator)
                if item is None:
                    return
                yield item

        try:
            assert self._service is not None, "TTS service not initialized"
            assert self._config is not None, "Synthesis configuration not created"

            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            logger.debug(f"{self}: Generating TTS [{text}]")

            responses = await asyncio.to_thread(read_audio_responses)

            async for resp in async_iterator(responses):
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=resp.audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                yield frame

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()
        except asyncio.TimeoutError:
            logger.error(f"{self} timeout waiting for audio response")
            yield ErrorFrame(error=f"{self} error: {e}")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
