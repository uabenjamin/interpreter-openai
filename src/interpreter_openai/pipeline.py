from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .audio_io import AudioUnavailableError, MicrophoneCapture, SpeakerPlayback, get_default_devices
from .config import AppConfig
from .error_handling import UserFacingError
from .openai_clients import build_client, verify_openai_text_generation
from .speech import OpenAITTSService
from .transcript_stream import OpenAIRealtimeTranscriber, TranscriptUpdate
from .translator import OpenAITranslator


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QueuedUtterance:
    sequence_id: int
    english_text: str


class InterpreterApp:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    async def run(self) -> None:
        client = build_client(self._config)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise UserFacingError(
                "OPENAI_API_KEY is not set. Save it in your shell profile, for example "
                "~/.zshrc, and restart the shell."
            )

        translator = OpenAITranslator(
            client=client,
            model=self._config.translation_model,
            max_output_tokens=self._config.translation_max_output_tokens,
            target_language_label=self._config.target_language_label,
            glossary_file=self._config.glossary_file,
            translation_notes_file=self._config.translation_notes_file,
        )
        tts = OpenAITTSService(
            client=client,
            model=self._config.tts_model,
            voice=self._config.tts_voice,
            instructions=self._config.tts_instructions,
            speed=self._config.tts_speed,
            sample_rate_hz=self._config.sample_rate_hz,
        )
        transcriber = OpenAIRealtimeTranscriber(self._config, api_key)
        speaker = SpeakerPlayback(
            sample_rate_hz=self._config.sample_rate_hz,
            drain_ms=self._config.playback_drain_ms,
            target_rms=self._config.playback_target_rms,
            max_gain=self._config.playback_max_gain,
        )

        devices = get_default_devices()
        LOGGER.info("Using microphone: %s", devices.default_microphone)
        LOGGER.info("Using speaker: %s", devices.default_speaker)
        LOGGER.info(
            "Realtime session model: %s | transcription model: %s",
            self._config.realtime_session_model,
            self._config.transcription_model,
        )
        LOGGER.info(
            "Turn detection: %s%s",
            self._config.turn_detection_type,
            (
                f" ({self._config.semantic_vad_eagerness})"
                if self._config.turn_detection_type == "semantic_vad"
                else ""
            ),
        )
        LOGGER.info("Max turn duration: %sms", self._config.max_turn_ms)
        LOGGER.info(
            "Translation buffer: silence=%sms max=%sms min_words=%s",
            self._config.translation_buffer_silence_ms,
            self._config.translation_buffer_max_ms,
            self._config.translation_min_words,
        )
        LOGGER.info(
            "Listening continuously. Use Control-C to stop. Command-C usually "
            "copies text and does not stop terminal apps on macOS."
        )
        if self._config.glossary_file:
            LOGGER.info("Glossary file: %s", self._config.glossary_file.expanduser())
        if self._config.translation_notes_file:
            LOGGER.info(
                "Translation notes file: %s",
                self._config.translation_notes_file.expanduser(),
            )

        transcript_queue: asyncio.Queue[TranscriptUpdate] = asyncio.Queue()
        utterance_queue: asyncio.Queue[QueuedUtterance | None] = asyncio.Queue()
        microphone = MicrophoneCapture(
            capture_sample_rate_hz=self._config.capture_sample_rate_hz,
            output_sample_rate_hz=self._config.sample_rate_hz,
            capture_chunk_frames=self._config.capture_chunk_frames,
        )

        async def audio_source() -> AsyncIterator[bytes]:
            async for chunk in microphone.chunks():
                yield chunk

        stream_ready = asyncio.Event()
        stream_task = asyncio.create_task(
            transcriber.stream_audio(audio_source(), transcript_queue, stream_ready)
        )
        playback_task = asyncio.create_task(
            self._translation_playback_worker(
                utterance_queue,
                translator,
                tts,
                speaker,
            )
        )

        try:
            await self._wait_for_stream_ready(stream_task, stream_ready)
            await microphone.start()
            await self._continuous_listen_loop(
                transcript_queue,
                utterance_queue,
                stream_task,
                playback_task,
            )
        finally:
            await microphone.stop()
            done, _ = await asyncio.wait({stream_task}, timeout=5)
            if stream_task in done:
                await stream_task
            else:
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            await utterance_queue.put(None)
            with contextlib.suppress(asyncio.CancelledError):
                await playback_task

    async def doctor(self) -> None:
        try:
            devices = get_default_devices()
        except AudioUnavailableError as exc:
            LOGGER.warning("Audio device check failed: %s", exc)
        else:
            LOGGER.info("Default microphone: %s", devices.default_microphone)
            LOGGER.info("Default speaker: %s", devices.default_speaker)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise UserFacingError(
                "OPENAI_API_KEY is not set. Save it in your shell profile, for example "
                "~/.zshrc, and restart the shell."
            )

        client = build_client(self._config)
        LOGGER.info(
            "Realtime session model: %s | transcription model: %s",
            self._config.realtime_session_model,
            self._config.transcription_model,
        )
        LOGGER.info(
            "Turn detection: %s%s",
            self._config.turn_detection_type,
            (
                f" ({self._config.semantic_vad_eagerness})"
                if self._config.turn_detection_type == "semantic_vad"
                else ""
            ),
        )
        LOGGER.info("Max turn duration: %sms", self._config.max_turn_ms)
        LOGGER.info(
            "Translation buffer: silence=%sms max=%sms min_words=%s",
            self._config.translation_buffer_silence_ms,
            self._config.translation_buffer_max_ms,
            self._config.translation_min_words,
        )
        translator_probe = await verify_openai_text_generation(
            client,
            self._config.translation_model,
        )
        LOGGER.info("OpenAI text probe OK: %s", translator_probe or "<empty>")

        tts = OpenAITTSService(
            client=client,
            model=self._config.tts_model,
            voice=self._config.tts_voice,
            instructions=self._config.tts_instructions,
            speed=self._config.tts_speed,
            sample_rate_hz=self._config.sample_rate_hz,
        )
        tts_bytes = await asyncio.to_thread(tts.verify_tts)
        LOGGER.info("OpenAI TTS probe OK: %s bytes", tts_bytes)

        transcriber = OpenAIRealtimeTranscriber(self._config, api_key)
        await transcriber.verify_connection()
        LOGGER.info("OpenAI realtime transcription probe OK.")

    async def _continuous_listen_loop(
        self,
        transcript_queue: asyncio.Queue[TranscriptUpdate],
        utterance_queue: asyncio.Queue[QueuedUtterance | None],
        stream_task: asyncio.Task[None],
        playback_task: asyncio.Task[None],
    ) -> None:
        last_partial_by_item: dict[str, str] = {}
        seen_completed_items: set[str] = set()
        next_sequence_id = 1
        buffered_english = ""
        buffer_started_at: float | None = None
        buffer_last_updated_at: float | None = None

        async def flush_buffered_english() -> None:
            nonlocal buffered_english, buffer_started_at, buffer_last_updated_at, next_sequence_id

            english_text = buffered_english.strip()
            if not english_text:
                buffered_english = ""
                buffer_started_at = None
                buffer_last_updated_at = None
                return

            if utterance_queue.qsize() >= 2:
                LOGGER.warning(
                    "Playback queue backlog is %s utterances. Mandarin audio may lag.",
                    utterance_queue.qsize(),
                )

            LOGGER.info("[segment-en %s] %s", next_sequence_id, english_text)
            await utterance_queue.put(
                QueuedUtterance(
                    sequence_id=next_sequence_id,
                    english_text=english_text,
                )
            )
            next_sequence_id += 1
            buffered_english = ""
            buffer_started_at = None
            buffer_last_updated_at = None

        while True:
            if playback_task.done():
                await playback_task
                raise RuntimeError("Playback worker stopped unexpectedly.")
            if stream_task.done():
                await stream_task
                raise RuntimeError("Realtime transcription stream ended unexpectedly.")

            try:
                update = await asyncio.wait_for(transcript_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                now = time.monotonic()
                if self._should_flush_translation_buffer(
                    buffered_english,
                    buffer_started_at,
                    buffer_last_updated_at,
                    now,
                ):
                    await flush_buffered_english()
                continue

            if update.is_partial:
                last_text = last_partial_by_item.get(update.item_id)
                if update.text != last_text:
                    last_partial_by_item[update.item_id] = update.text
                    LOGGER.info("[partial-en] %s", update.text)
                continue

            if update.item_id in seen_completed_items:
                continue
            seen_completed_items.add(update.item_id)
            last_partial_by_item.pop(update.item_id, None)
            english_text = update.text.strip()
            if not english_text:
                continue

            now = time.monotonic()
            normalized_fragment = self._normalize_transcript_fragment(english_text)
            if not normalized_fragment:
                continue
            buffered_english = self._merge_transcript_fragments(
                buffered_english,
                normalized_fragment,
            )
            if buffer_started_at is None:
                buffer_started_at = now
            buffer_last_updated_at = now

            if self._should_flush_translation_buffer(
                buffered_english,
                buffer_started_at,
                buffer_last_updated_at,
                now,
            ):
                await flush_buffered_english()

    async def _translation_playback_worker(
        self,
        utterance_queue: asyncio.Queue[QueuedUtterance | None],
        translator: OpenAITranslator,
        tts: OpenAITTSService,
        speaker: SpeakerPlayback,
    ) -> None:
        while True:
            utterance = await utterance_queue.get()
            if utterance is None:
                return

            try:
                LOGGER.info("[final-en %s] %s", utterance.sequence_id, utterance.english_text)

                translate_started = time.monotonic()
                mandarin_text = await translator.translate_text(utterance.english_text)
                translate_elapsed_ms = (time.monotonic() - translate_started) * 1000
                LOGGER.info("[zh %s] %s", utterance.sequence_id, mandarin_text)

                tts_metrics = await tts.stream_to_speaker(mandarin_text, speaker)
                LOGGER.info(
                    "Latencies[%s]: translate=%.0fms tts_first_audio=%.0fms tts_total=%.0fms",
                    utterance.sequence_id,
                    translate_elapsed_ms,
                    tts_metrics.first_audio_ms or -1.0,
                    tts_metrics.total_ms,
                )
            except Exception:
                LOGGER.exception(
                    "Failed while processing utterance %s.",
                    utterance.sequence_id,
                )
            finally:
                utterance_queue.task_done()

    def _normalize_transcript_fragment(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _merge_transcript_fragments(self, existing: str, fragment: str) -> str:
        if not existing:
            return fragment
        if not fragment:
            return existing

        existing_lower = existing.lower()
        fragment_lower = fragment.lower()
        if existing_lower.endswith(fragment_lower):
            return existing

        max_overlap = min(len(existing), len(fragment))
        for overlap in range(max_overlap, 0, -1):
            if existing_lower.endswith(fragment_lower[:overlap]):
                tail = fragment[overlap:]
                if not tail:
                    return existing
                if tail[0] in " ,.!?;:)":
                    return f"{existing}{tail}"
                return f"{existing} {tail}"

        if existing[-1] in " ([{" or fragment[0] in ",.!?;:)]}":
            return f"{existing}{fragment}"
        return f"{existing} {fragment}"

    def _should_flush_translation_buffer(
        self,
        buffered_english: str,
        buffer_started_at: float | None,
        buffer_last_updated_at: float | None,
        now: float,
    ) -> bool:
        text = buffered_english.strip()
        if not text or buffer_started_at is None or buffer_last_updated_at is None:
            return False

        word_count = self._word_count(text)
        idle_ms = (now - buffer_last_updated_at) * 1000
        age_ms = (now - buffer_started_at) * 1000

        if self._ends_with_sentence_punctuation(text):
            return True
        if (
            word_count >= self._config.translation_min_words
            and idle_ms >= self._config.translation_buffer_silence_ms
        ):
            return True
        if (
            word_count >= self._config.translation_min_words
            and age_ms >= self._config.translation_buffer_max_ms
        ):
            return True
        if idle_ms >= max(self._config.translation_buffer_silence_ms * 2, 1500):
            return True
        return False

    def _ends_with_sentence_punctuation(self, text: str) -> bool:
        return bool(re.search(r"[.!?。！？][\"')\]]*$", text)) or text.endswith("...")

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b[\w']+\b", text))

    async def _wait_for_stream_ready(
        self,
        stream_task: asyncio.Task[None],
        ready_event: asyncio.Event,
    ) -> None:
        while not ready_event.is_set():
            if stream_task.done():
                await stream_task
                raise RuntimeError(
                    "Realtime transcription stream ended before it became ready."
                )
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
