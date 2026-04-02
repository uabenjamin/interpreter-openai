from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from .config import AppConfig
from .error_handling import UserFacingError


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptUpdate:
    item_id: str
    text: str
    is_partial: bool


class OpenAIRealtimeTranscriber:
    def __init__(self, config: AppConfig, api_key: str) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=api_key,
            project=config.openai_project,
        )
        self._commit_counter = 0

    async def stream_audio(
        self,
        audio_source: AsyncIterator[bytes],
        transcript_queue: asyncio.Queue[TranscriptUpdate],
        ready_event: asyncio.Event,
    ) -> None:
        async with self._client.beta.realtime.connect(
            model=self._config.realtime_session_model,
            websocket_connection_options={
                "max_size": None,
                "ping_interval": 20,
                "ping_timeout": 20,
            },
        ) as connection:
            await connection.session.update(session=self._session_update())
            receiver = asyncio.create_task(
                self._receive_events(connection, transcript_queue, ready_event)
            )
            sender = asyncio.create_task(self._send_audio(connection, audio_source))

            done, pending = await asyncio.wait(
                {receiver, sender},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
            for task in done:
                await task
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def verify_connection(self) -> None:
        async with self._client.beta.realtime.connect(
            model=self._config.realtime_session_model,
            websocket_connection_options={
                "max_size": None,
                "ping_interval": 20,
                "ping_timeout": 20,
            },
        ) as connection:
            await connection.session.update(session=self._session_update())
            deadline = asyncio.get_running_loop().time() + 10
            while True:
                if asyncio.get_running_loop().time() > deadline:
                    raise UserFacingError(
                        "Timed out while opening the OpenAI realtime transcription session."
                    )
                event = await asyncio.wait_for(connection.recv(), timeout=1)
                event_type = getattr(event, "type", None)
                if event_type == "session.created":
                    continue
                if event_type == "session.updated":
                    return
                if event_type == "error":
                    raise UserFacingError(self._format_error_event(event))

    def _session_update(self) -> dict[str, object]:
        session: dict[str, object] = {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": self._config.transcription_model,
                "language": self._config.source_language,
                "prompt": (
                    "Transcribe spoken English clearly. Keep scripture references, "
                    "proper names, and church terminology accurate."
                ),
            },
            "modalities": ["text"],
            "turn_detection": {
                "type": "server_vad",
                "threshold": self._config.vad_threshold,
                "prefix_padding_ms": self._config.vad_prefix_padding_ms,
                "silence_duration_ms": self._config.vad_silence_ms,
                "create_response": False,
                "interrupt_response": False,
            },
        }
        if self._config.noise_reduction_mode is not None:
            session["input_audio_noise_reduction"] = {
                "type": self._config.noise_reduction_mode
            }
        return session

    async def _send_audio(
        self,
        connection,
        audio_source: AsyncIterator[bytes],
    ) -> None:
        chunks_sent = 0
        async for chunk in audio_source:
            await connection.input_audio_buffer.append(
                audio=base64.b64encode(chunk).decode("ascii")
            )
            chunks_sent += 1
            if chunks_sent == 1:
                LOGGER.info("Streaming microphone audio to OpenAI.")

    async def _receive_events(
        self,
        connection,
        transcript_queue: asyncio.Queue[TranscriptUpdate],
        ready_event: asyncio.Event,
    ) -> None:
        partial_text_by_item: dict[str, str] = {}
        pending_commits: dict[str, str | None] = {}
        commit_index: dict[str, int] = {}
        completed_text_by_item: dict[str, str] = {}
        emitted_items: set[str] = set()

        async for event in connection:
            event_type = getattr(event, "type", None)

            if event_type == "session.created":
                LOGGER.info("OpenAI realtime session created.")
                continue

            if event_type == "session.updated":
                LOGGER.info("OpenAI realtime session configured for transcription.")
                ready_event.set()
                continue

            if event_type == "input_audio_buffer.speech_started":
                LOGGER.info("Speech detected.")
                continue

            if event_type == "input_audio_buffer.speech_stopped":
                LOGGER.info("Speech ended. Waiting for transcription.")
                continue

            if event_type == "input_audio_buffer.committed":
                item_id = str(getattr(event, "item_id", "") or "")
                if not item_id:
                    continue
                previous_item_id = getattr(event, "previous_item_id", None)
                LOGGER.info("Audio turn committed.")
                pending_commits[item_id] = (
                    str(previous_item_id) if previous_item_id is not None else None
                )
                self._commit_counter += 1
                commit_index[item_id] = self._commit_counter
                await self._emit_ready_completed(
                    transcript_queue,
                    pending_commits,
                    commit_index,
                    completed_text_by_item,
                    emitted_items,
                )
                continue

            if event_type == "conversation.item.input_audio_transcription.delta":
                item_id = str(getattr(event, "item_id", "") or "")
                delta = str(getattr(event, "delta", "") or "")
                if not item_id or not delta:
                    continue
                next_text = f"{partial_text_by_item.get(item_id, '')}{delta}"
                partial_text_by_item[item_id] = next_text
                await transcript_queue.put(
                    TranscriptUpdate(item_id=item_id, text=next_text, is_partial=True)
                )
                continue

            if event_type == "conversation.item.input_audio_transcription.segment":
                item_id = str(getattr(event, "item_id", "") or "")
                segment_text = str(getattr(event, "text", "") or "").strip()
                if not item_id or not segment_text:
                    continue
                await transcript_queue.put(
                    TranscriptUpdate(item_id=item_id, text=segment_text, is_partial=True)
                )
                continue

            if event_type == "conversation.item.input_audio_transcription.completed":
                item_id = str(getattr(event, "item_id", "") or "")
                transcript = str(getattr(event, "transcript", "") or "").strip()
                if not item_id:
                    continue
                if transcript:
                    completed_text_by_item[item_id] = transcript
                partial_text_by_item.pop(item_id, None)
                await self._emit_ready_completed(
                    transcript_queue,
                    pending_commits,
                    commit_index,
                    completed_text_by_item,
                    emitted_items,
                )
                continue

            if event_type == "conversation.item.input_audio_transcription.failed":
                error = getattr(event, "error", None)
                message = getattr(error, "message", None) or "Unknown transcription failure."
                LOGGER.warning("Input audio transcription failed: %s", message)
                continue

            if event_type == "error":
                raise UserFacingError(self._format_error_event(event))

    async def _emit_ready_completed(
        self,
        transcript_queue: asyncio.Queue[TranscriptUpdate],
        pending_commits: dict[str, str | None],
        commit_index: dict[str, int],
        completed_text_by_item: dict[str, str],
        emitted_items: set[str],
    ) -> None:
        while True:
            ready_items = [
                item_id
                for item_id, transcript in completed_text_by_item.items()
                if transcript
                and item_id not in emitted_items
                and (
                    pending_commits.get(item_id) is None
                    or pending_commits.get(item_id) in emitted_items
                )
            ]
            if not ready_items:
                return

            next_item_id = min(
                ready_items,
                key=lambda item_id: commit_index.get(item_id, 1_000_000_000),
            )
            emitted_items.add(next_item_id)
            transcript = completed_text_by_item.pop(next_item_id).strip()
            if transcript:
                await transcript_queue.put(
                    TranscriptUpdate(
                        item_id=next_item_id,
                        text=transcript,
                        is_partial=False,
                    )
                )

    def _format_error_event(self, event) -> str:
        error = getattr(event, "error", None)
        message = getattr(error, "message", None)
        if message:
            return f"OpenAI realtime error: {message}"
        return f"OpenAI realtime error: {event}"
