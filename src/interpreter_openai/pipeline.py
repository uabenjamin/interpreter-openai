from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from .audio_io import (
    AudioFileCapture,
    AudioUnavailableError,
    MicrophoneCapture,
    SpeakerPlayback,
    SpeechFilterConfig,
    get_default_microphone_name,
    get_default_speaker_name,
    get_selected_microphone_name,
    get_selected_speaker_name,
    list_microphone_names,
    list_speaker_names,
)
from .config import AppConfig
from .error_handling import UserFacingError
from .openai_clients import build_client, verify_openai_text_generation
from .speech import OpenAITTSService
from .transcript_stream import OpenAIRealtimeTranscriber, TranscriptUpdate
from .translator import OpenAITranslator


LOGGER = logging.getLogger(__name__)
INCOMPLETE_TRAILING_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "because",
    "before",
    "both",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "like",
    "nor",
    "of",
    "or",
    "so",
    "that",
    "the",
    "through",
    "to",
    "toward",
    "towards",
    "until",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "with",
    "without",
}
INCOMPLETE_TRAILING_PHRASES = (
    "as we",
    "both for",
    "called to",
    "come and",
    "come a day when",
    "every mouth will",
    "every tongue",
    "every tongue every mouth",
    "get into",
    "give you some",
    "he is who he",
    "if you are interested",
    "in order to",
    "is about",
    "is who he",
    "it is",
    "it's",
    "not only",
    "no voice",
    "pray for that",
    "pray that",
    "so that",
    "the meaning of",
    "those of",
    "those who",
    "to the",
    "we are to",
    "we can",
    "we don't see",
    "we do not see",
    "we give you",
    "we need to",
    "we pray that",
    "we want to",
    "we're going to",
    "what we need to hear",
    "will be no voice",
    "you can",
    "you need to",
)
INCOMPLETE_TRAILING_PATTERNS = (
    re.compile(r"\b(as|when|while|before|after|if|because|that)\s+we$", re.IGNORECASE),
    re.compile(r"\b(all|some|many|those|the|a|an)\s+\w+$", re.IGNORECASE),
    re.compile(r"\b(we|you|they|he|she|it|there)\s+(will|would|can|could|may|might|must|should|have|has|are|is|were|was|do|does|did)\s+\w+$", re.IGNORECASE),
    re.compile(r"\b(will|would|can|could|may|might|must|should|to|for|with|into|about|from|upon)\s+\w+$", re.IGNORECASE),
)


def _emit_console_line(label: str, sequence_id: int, text: str) -> None:
    print(f"[{label} {sequence_id}] {text}", flush=True)


def _emit_status_line(text: str) -> None:
    print(f"[status] {text}", flush=True)


@dataclass(slots=True)
class QueuedUtterance:
    sequence_id: int
    english_text: str


@dataclass(slots=True)
class TranslatedUtterance:
    sequence_id: int
    translated_text: str
    translate_elapsed_ms: float


class InterpreterApp:
    def __init__(
        self,
        config: AppConfig,
        output_handler: Callable[[str, int | None, str], None] | None = None,
        status_handler: Callable[[str], None] | None = None,
        sermon_reference_text: str | None = None,
    ) -> None:
        self._config = config
        self._output_handler = output_handler
        self._status_handler = status_handler
        self._sermon_reference_text = sermon_reference_text

    def _emit_console_line(self, label: str, sequence_id: int, text: str) -> None:
        if self._output_handler is not None:
            self._output_handler(label, sequence_id, text)
            return
        _emit_console_line(label, sequence_id, text)

    def _emit_status_line(self, text: str) -> None:
        if self._status_handler is not None:
            self._status_handler(text)
            return
        _emit_status_line(text)

    async def list_devices(self) -> None:
        print("Input devices:", flush=True)
        try:
            default_microphone = get_default_microphone_name()
            for name in list_microphone_names():
                marker = "*" if name == default_microphone else "-"
                print(f"  {marker} {name}", flush=True)
        except AudioUnavailableError as exc:
            print(f"  unavailable: {exc}", flush=True)

        print("Output devices:", flush=True)
        try:
            default_speaker = get_default_speaker_name()
            for name in list_speaker_names():
                marker = "*" if name == default_speaker else "-"
                print(f"  {marker} {name}", flush=True)
        except AudioUnavailableError as exc:
            print(f"  unavailable: {exc}", flush=True)

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
            sermon_reference_text=self._sermon_reference_text,
        )
        transcriber = OpenAIRealtimeTranscriber(self._config, api_key)
        tts: OpenAITTSService | None = None
        speaker: SpeakerPlayback | None = None
        if self._config.enable_tts:
            tts = OpenAITTSService(
                client=client,
                model=self._config.tts_model,
                voice=self._config.tts_voice,
                instructions=self._config.tts_instructions,
                speed=self._config.tts_speed,
                sample_rate_hz=self._config.sample_rate_hz,
            )
            speaker = SpeakerPlayback(
                device_name=self._config.output_device,
                sample_rate_hz=self._config.sample_rate_hz,
                drain_ms=self._config.playback_drain_ms,
                target_rms=self._config.playback_target_rms,
                max_gain=self._config.playback_max_gain,
            )

        if self._config.input_audio_file is not None:
            input_source_label = str(self._config.input_audio_file)
            LOGGER.info("Using input audio file: %s", input_source_label)
            self._emit_status_line(f"input audio file: {input_source_label}")
        else:
            try:
                microphone_name = get_selected_microphone_name(self._config.input_device)
            except AudioUnavailableError as exc:
                raise UserFacingError(str(exc)) from exc
            input_source_label = microphone_name
            LOGGER.info("Using microphone: %s", microphone_name)
            self._emit_status_line(f"input device: {microphone_name}")
        if self._config.enable_tts:
            try:
                speaker_name = get_selected_speaker_name(self._config.output_device)
            except AudioUnavailableError as exc:
                raise UserFacingError(str(exc)) from exc
            LOGGER.info("Using speaker: %s", speaker_name)
            self._emit_status_line(f"output device: {speaker_name}")
        else:
            if self._config.output_device:
                self._emit_status_line(
                    "TTS is disabled, so --output-device is ignored. Add --enable-tts for speaker output."
                )
            else:
                self._emit_status_line("TTS is disabled; translated text will print to stdout only.")
        LOGGER.info("Realtime transcription model: %s", self._config.transcription_model)
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
        LOGGER.info("Speech filter: %s", self._config.speech_filter_mode)
        LOGGER.info("TTS: %s", "enabled" if self._config.enable_tts else "disabled")
        LOGGER.info(
            "Listening continuously. Use Control-C to stop. Command-C usually "
            "copies text and does not stop terminal apps on macOS."
        )
        self._emit_status_line(
            "connecting to OpenAI realtime transcription; final output appears as [en ...] and [target ...]."
        )
        if self._config.max_turn_ms > 0:
            self._emit_status_line(
                f"manual audio commit interval: {self._config.max_turn_ms / 1000:.1f}s."
            )
        if self._config.glossary_file:
            LOGGER.info("Glossary file: %s", self._config.glossary_file.expanduser())
        if self._config.translation_notes_file:
            LOGGER.info(
                "Translation notes file: %s",
                self._config.translation_notes_file.expanduser(),
            )
        if self._sermon_reference_text:
            self._emit_status_line("sermon reference loaded for translation.")

        transcript_queue: asyncio.Queue[TranscriptUpdate] = asyncio.Queue()
        utterance_queue: asyncio.Queue[QueuedUtterance | None] = asyncio.Queue()
        translated_audio_queue: asyncio.Queue[TranslatedUtterance | None] | None = None
        if tts is not None and speaker is not None:
            translated_audio_queue = asyncio.Queue()
        speech_filter_config = SpeechFilterConfig(
            mode=self._config.speech_filter_mode,
            sample_rate_hz=self._config.sample_rate_hz,
            highpass_hz=self._config.speech_filter_highpass_hz,
            lowpass_hz=self._config.speech_filter_lowpass_hz,
            gate_threshold=self._config.speech_filter_gate_threshold,
            gate_floor=self._config.speech_filter_gate_floor,
        )
        if self._config.input_audio_file is not None:
            input_source = AudioFileCapture(
                path=self._config.input_audio_file,
                output_sample_rate_hz=self._config.sample_rate_hz,
                output_chunk_frames=self._config.chunk_frames,
                speech_filter_config=speech_filter_config,
            )
        else:
            input_source = MicrophoneCapture(
                device_name=self._config.input_device,
                capture_sample_rate_hz=self._config.capture_sample_rate_hz,
                output_sample_rate_hz=self._config.sample_rate_hz,
                capture_chunk_frames=self._config.capture_chunk_frames,
                speech_filter_config=speech_filter_config,
            )

        async def audio_source() -> AsyncIterator[bytes]:
            async for chunk in input_source.chunks():
                yield chunk

        stream_ready = asyncio.Event()
        stream_task = asyncio.create_task(
            transcriber.stream_audio(audio_source(), transcript_queue, stream_ready)
        )
        translation_task = asyncio.create_task(
            self._translation_worker(
                utterance_queue,
                translator,
                translated_audio_queue,
            )
        )
        audio_playback_task: asyncio.Task[None] | None = None
        if translated_audio_queue is not None and tts is not None and speaker is not None:
            audio_playback_task = asyncio.create_task(
                self._translated_audio_playback_worker(
                    translated_audio_queue,
                    tts,
                    speaker,
                )
            )
        worker_tasks = {translation_task}
        if audio_playback_task is not None:
            worker_tasks.add(audio_playback_task)

        try:
            await self._wait_for_stream_ready(stream_task, stream_ready)
            await input_source.start()
            self._emit_status_line(
                "streaming input now; final output appears as [en ...] and [target ...]."
            )
            await self._continuous_listen_loop(
                transcript_queue,
                utterance_queue,
                stream_task,
                worker_tasks,
            )
        finally:
            await input_source.stop()
            done, _ = await asyncio.wait({stream_task}, timeout=5)
            if stream_task in done:
                await stream_task
            else:
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            await utterance_queue.put(None)
            with contextlib.suppress(asyncio.CancelledError):
                await translation_task
            if audio_playback_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await audio_playback_task

    async def doctor(self) -> None:
        if self._config.input_audio_file is not None:
            if self._config.input_audio_file.exists() and self._config.input_audio_file.is_file():
                LOGGER.info("Input audio file: %s", self._config.input_audio_file)
            else:
                LOGGER.warning("Input audio file is not readable: %s", self._config.input_audio_file)
        else:
            try:
                microphone_name = get_selected_microphone_name(self._config.input_device)
            except AudioUnavailableError as exc:
                LOGGER.warning("Audio device check failed: %s", exc)
            else:
                label = "Selected microphone" if self._config.input_device else "Default microphone"
                LOGGER.info("%s: %s", label, microphone_name)
        if self._config.enable_tts:
            try:
                speaker_name = get_selected_speaker_name(self._config.output_device)
            except AudioUnavailableError as exc:
                LOGGER.warning("Speaker check failed: %s", exc)
            else:
                label = "Selected speaker" if self._config.output_device else "Default speaker"
                LOGGER.info("%s: %s", label, speaker_name)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise UserFacingError(
                "OPENAI_API_KEY is not set. Save it in your shell profile, for example "
                "~/.zshrc, and restart the shell."
            )

        client = build_client(self._config)
        LOGGER.info("Realtime transcription model: %s", self._config.transcription_model)
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
        LOGGER.info("Speech filter: %s", self._config.speech_filter_mode)
        LOGGER.info("TTS: %s", "enabled" if self._config.enable_tts else "disabled")
        translator_probe = await verify_openai_text_generation(
            client,
            self._config.translation_model,
        )
        LOGGER.info("OpenAI text probe OK: %s", translator_probe or "<empty>")

        if self._config.enable_tts:
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
        else:
            LOGGER.info("OpenAI TTS probe skipped because TTS is disabled.")

        transcriber = OpenAIRealtimeTranscriber(self._config, api_key)
        await transcriber.verify_connection()
        LOGGER.info("OpenAI realtime transcription probe OK.")

    async def _continuous_listen_loop(
        self,
        transcript_queue: asyncio.Queue[TranscriptUpdate],
        utterance_queue: asyncio.Queue[QueuedUtterance | None],
        stream_task: asyncio.Task[None],
        worker_tasks: set[asyncio.Task[None]],
    ) -> None:
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
                    "Translation queue backlog is %s utterances. Translated output may lag.",
                    utterance_queue.qsize(),
                )

            self._emit_console_line("en", next_sequence_id, english_text)
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
            stopped_workers = [task for task in worker_tasks if task.done()]
            if stopped_workers:
                await stopped_workers[0]
                raise RuntimeError("Translation or playback worker stopped unexpectedly.")
            if stream_task.done() and self._config.input_audio_file is not None:
                await stream_task
                if not transcript_queue.empty():
                    pass
                else:
                    await flush_buffered_english()
                    await utterance_queue.join()
                    return
            elif stream_task.done():
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
                continue

            if update.item_id in seen_completed_items:
                continue
            seen_completed_items.add(update.item_id)
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

    async def _translation_worker(
        self,
        utterance_queue: asyncio.Queue[QueuedUtterance | None],
        translator: OpenAITranslator,
        translated_audio_queue: asyncio.Queue[TranslatedUtterance | None] | None,
    ) -> None:
        while True:
            utterance = await utterance_queue.get()
            if utterance is None:
                if translated_audio_queue is not None:
                    await translated_audio_queue.put(None)
                utterance_queue.task_done()
                return

            try:
                translate_started = time.monotonic()
                translated_text = await translator.translate_text(utterance.english_text)
                translate_elapsed_ms = (time.monotonic() - translate_started) * 1000
                self._emit_console_line("target", utterance.sequence_id, translated_text)

                if translated_audio_queue is not None:
                    await translated_audio_queue.put(
                        TranslatedUtterance(
                            sequence_id=utterance.sequence_id,
                            translated_text=translated_text,
                            translate_elapsed_ms=translate_elapsed_ms,
                        )
                    )
                    if translated_audio_queue.qsize() >= 2:
                        LOGGER.warning(
                            "TTS playback queue backlog is %s utterances. Translated audio may lag.",
                            translated_audio_queue.qsize(),
                        )
                else:
                    LOGGER.info(
                        "Latencies[%s]: translate=%.0fms",
                        utterance.sequence_id,
                        translate_elapsed_ms,
                    )
            except Exception:
                LOGGER.exception(
                    "Failed while translating utterance %s.",
                    utterance.sequence_id,
                )
            finally:
                utterance_queue.task_done()

    async def _translated_audio_playback_worker(
        self,
        translated_audio_queue: asyncio.Queue[TranslatedUtterance | None],
        tts: OpenAITTSService,
        speaker: SpeakerPlayback,
    ) -> None:
        while True:
            utterance = await translated_audio_queue.get()
            if utterance is None:
                translated_audio_queue.task_done()
                return

            try:
                tts_metrics = await tts.stream_to_speaker(utterance.translated_text, speaker)
                LOGGER.info(
                    "Latencies[%s]: translate=%.0fms tts_first_audio=%.0fms tts_total=%.0fms",
                    utterance.sequence_id,
                    utterance.translate_elapsed_ms,
                    tts_metrics.first_audio_ms or -1.0,
                    tts_metrics.total_ms,
                )
            except Exception:
                LOGGER.exception(
                    "Failed while playing translated audio for utterance %s.",
                    utterance.sequence_id,
                )
            finally:
                translated_audio_queue.task_done()

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

        if self._looks_like_incomplete_clause(text):
            extended_idle_ms = max(self._config.translation_buffer_silence_ms * 4, 3500)
            extended_age_ms = max(
                int(self._config.translation_buffer_max_ms * 1.5),
                self._config.translation_buffer_max_ms + self._config.max_turn_ms,
            )
            if word_count >= self._config.translation_min_words and age_ms >= extended_age_ms:
                return True
            if idle_ms >= extended_idle_ms:
                return True
            return False

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

    def _looks_like_incomplete_clause(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith((",", ";", ":", "-", "–", "—")):
            return True

        normalized = re.sub(r"[^a-z0-9'\s]+$", "", stripped.lower()).strip()
        if not normalized:
            return False
        words = re.findall(r"\b[\w']+\b", normalized)
        if not words:
            return False
        if words[-1] in INCOMPLETE_TRAILING_WORDS:
            return True
        if any(normalized.endswith(phrase) for phrase in INCOMPLETE_TRAILING_PHRASES):
            return True
        return any(pattern.search(normalized) for pattern in INCOMPLETE_TRAILING_PATTERNS)

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
