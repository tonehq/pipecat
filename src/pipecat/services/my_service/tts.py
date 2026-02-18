from pipecat.services.tts_service import TTSService
from openai import AsyncOpenAI

class MyTTSService(TTSService):
      def __init__(self, api_key: str, voice: str = "alloy"):
            self.api_key = api_key
            self.voice = voice


      async def run_tts(self, text: str) -> bytes:
            client = AsyncOpenAI(api_key=self.api_key)

            response = await client.audio.speech.create(
                  model="gpt-4o-mini-tts",
                  voice=self.voice,
                  input=text
            )

            return response.read()

      def get_voices(self):
        return [
            {
                "voice_id": "default",
                "name": "Default Voice"
            }
        ]



