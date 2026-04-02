from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from openai import OpenAI

from .audio_io import SpeakerPlayback


@dataclass(slots=True)
class SynthesizedSpeech:
    audio_bytes: bytes
    sample_rate_hz: int


@dataclass(slots=True)
class TTSPlaybackMetrics:
    first_audio_ms: float | None
    total_ms: float


class OpenAITTSService:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        voice: str,
        instructions: str,
        speed: float,
        sample_rate_hz: int,
    ) -> None:
        self._client = client
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._speed = speed
        self._sample_rate_hz = sample_rate_hz

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        return await asyncio.to_thread(self._synthesize_blocking, text)

    async def stream_to_speaker(
        self,
        text: str,
        speaker: SpeakerPlayback,
    ) -> TTSPlaybackMetrics:
        return await asyncio.to_thread(self._stream_to_speaker_blocking, text, speaker)

    def verify_tts(self) -> int:
        return len(self._synthesize_blocking("平安。").audio_bytes)

    def _synthesize_blocking(self, text: str) -> SynthesizedSpeech:
        with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            instructions=self._instructions,
            response_format="pcm",
            speed=self._speed,
        ) as response:
            audio_bytes = b"".join(response.iter_bytes())
        if not audio_bytes:
            raise RuntimeError("OpenAI TTS returned no audio bytes.")
        return SynthesizedSpeech(
            audio_bytes=audio_bytes,
            sample_rate_hz=self._sample_rate_hz,
        )

    def _stream_to_speaker_blocking(
        self,
        text: str,
        speaker: SpeakerPlayback,
    ) -> TTSPlaybackMetrics:
        started = time.monotonic()
        first_audio_ms: list[float | None] = [None]

        with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            instructions=self._instructions,
            response_format="pcm",
            speed=self._speed,
        ) as response:
            def iter_audio_chunks():
                for chunk in response.iter_bytes():
                    if chunk and first_audio_ms[0] is None:
                        first_audio_ms[0] = (time.monotonic() - started) * 1000
                    yield chunk

            speaker.play_pcm16_stream_blocking(iter_audio_chunks())

        return TTSPlaybackMetrics(
            first_audio_ms=first_audio_ms[0],
            total_ms=(time.monotonic() - started) * 1000,
        )
