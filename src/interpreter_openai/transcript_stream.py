from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from .config import AppConfig
from .error_handling import UserFacingError


LOGGER = logging.getLogger(__name__)
TRANSCRIPTION_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
MANUAL_COMMIT_TIMEOUT_SECONDS = 8.0


@dataclass(slots=True)
class TranscriptUpdate:
    item_id: str
    text: str
    is_partial: bool


class OpenAIRealtimeTranscriber:
    def __init__(self, config: AppConfig, api_key: str) -> None:
        self._config = config
        self._api_key = api_key
        self._commit_counter = 0
        self._speech_active = False
        self._turn_started_at: float | None = None
        self._manual_commit_pending = False
        self._manual_commit_sent_at: float | None = None
        self._manual_commit_timeout_warning_emitted = False
        self._buffered_chunk_count = 0

    async def stream_audio(
        self,
        audio_source: AsyncIterator[bytes],
        transcript_queue: asyncio.Queue[TranscriptUpdate],
        ready_event: asyncio.Event,
    ) -> None:
        self._speech_active = False
        self._turn_started_at = None
        self._manual_commit_pending = False
        self._manual_commit_sent_at = None
        self._manual_commit_timeout_warning_emitted = False
        self._buffered_chunk_count = 0
        async with self._connect_transcription() as connection:
            await self._send_event(connection, self._transcription_session_update())
            receiver = asyncio.create_task(
                self._receive_events(connection, transcript_queue, ready_event)
            )
            sender = asyncio.create_task(self._send_audio(connection, audio_source))
            tasks = {receiver, sender}
            if self._config.max_turn_ms > 0:
                tasks.add(asyncio.create_task(self._force_commit_long_turns(connection)))

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if sender in done and sender.exception() is None:
                await sender
                await asyncio.sleep(max(3.0, (self._config.max_turn_ms / 1000) + 2.0))
                for task in pending:
                    task.cancel()
                for task in pending:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                return
            for task in pending:
                task.cancel()
            for task in done:
                await task
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def verify_connection(self) -> None:
        async with self._connect_transcription() as connection:
            await self._send_event(connection, self._transcription_session_update())
            deadline = asyncio.get_running_loop().time() + 10
            while True:
                if asyncio.get_running_loop().time() > deadline:
                    raise UserFacingError(
                        "Timed out while opening the OpenAI realtime transcription session."
                    )
                try:
                    event = await asyncio.wait_for(
                        self._recv_event(connection),
                        timeout=1,
                    )
                except asyncio.TimeoutError:
                    continue
                except ConnectionClosed as exc:
                    raise UserFacingError(
                        "OpenAI realtime transcription session closed before it was configured."
                    ) from exc
                event_type = self._event_field(event, "type")
                if event_type in {"session.created", "transcription_session.created"}:
                    continue
                if event_type in {"session.updated", "transcription_session.updated"}:
                    return
                if event_type == "error":
                    raise UserFacingError(self._format_error_event(event))

    def _connect_transcription(self):
        return connect(
            TRANSCRIPTION_WEBSOCKET_URL,
            additional_headers=self._auth_headers(),
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._config.openai_project:
            headers["OpenAI-Project"] = self._config.openai_project
        return headers

    async def _send_event(
        self,
        connection: ClientConnection,
        event: dict[str, object],
    ) -> None:
        await connection.send(json.dumps(event))

    async def _iter_events(
        self,
        connection: ClientConnection,
    ) -> AsyncIterator[dict[str, object]]:
        async for message in connection:
            yield self._parse_event_message(message)

    async def _recv_event(self, connection: ClientConnection) -> dict[str, object]:
        message = await connection.recv()
        return self._parse_event_message(message)

    def _parse_event_message(self, message: str | bytes) -> dict[str, object]:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        parsed = json.loads(message)
        if not isinstance(parsed, dict):
            return {"type": "unknown", "payload": parsed}
        return parsed

    def _transcription_session_update(self) -> dict[str, object]:
        audio_input: dict[str, object] = {
            "format": {
                "type": "audio/pcm",
                "rate": self._config.sample_rate_hz,
            },
            "transcription": {
                "model": self._config.transcription_model,
                "language": self._config.source_language,
            },
        }
        turn_detection = self._build_turn_detection()
        if turn_detection is not None:
            audio_input["turn_detection"] = turn_detection

        session: dict[str, object] = {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": audio_input,
                },
            },
        }
        if self._config.noise_reduction_mode is not None:
            session["session"]["audio"]["input"]["noise_reduction"] = {
                "type": self._config.noise_reduction_mode
            }
        return session

    def _build_turn_detection(self) -> dict[str, object] | None:
        if (
            self._config.turn_detection_type == "none"
            or self._config.transcription_model == "gpt-realtime-whisper"
        ):
            return None
        if self._config.turn_detection_type == "semantic_vad":
            return {
                "type": "semantic_vad",
                "eagerness": self._config.semantic_vad_eagerness,
            }
        return {
            "type": "server_vad",
            "threshold": self._config.vad_threshold,
            "prefix_padding_ms": self._config.vad_prefix_padding_ms,
            "silence_duration_ms": self._config.vad_silence_ms,
        }

    async def _send_audio(
        self,
        connection: ClientConnection,
        audio_source: AsyncIterator[bytes],
    ) -> None:
        chunks_sent = 0
        async for chunk in audio_source:
            await self._send_event(
                connection,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                },
            )
            chunks_sent += 1
            self._buffered_chunk_count += 1
            if not self._uses_server_turn_detection() and self._turn_started_at is None:
                self._turn_started_at = asyncio.get_running_loop().time()
            if chunks_sent == 1:
                LOGGER.info("Streaming microphone audio to OpenAI.")

        if chunks_sent > 0 and self._buffered_audio_ms() >= 250 and not self._manual_commit_pending:
            self._manual_commit_pending = True
            self._manual_commit_sent_at = asyncio.get_running_loop().time()
            await self._send_event(connection, {"type": "input_audio_buffer.commit"})

    async def _receive_events(
        self,
        connection: ClientConnection,
        transcript_queue: asyncio.Queue[TranscriptUpdate],
        ready_event: asyncio.Event,
    ) -> None:
        partial_text_by_item: dict[str, str] = {}
        pending_commits: dict[str, str | None] = {}
        commit_index: dict[str, int] = {}
        completed_text_by_item: dict[str, str] = {}
        emitted_items: set[str] = set()

        async for event in self._iter_events(connection):
            event_type = self._event_field(event, "type")

            if event_type in {"session.created", "transcription_session.created"}:
                LOGGER.info("OpenAI realtime session created.")
                continue

            if event_type in {"session.updated", "transcription_session.updated"}:
                LOGGER.info("OpenAI realtime session configured for transcription.")
                ready_event.set()
                continue

            if event_type == "input_audio_buffer.speech_started":
                self._speech_active = True
                if self._turn_started_at is None:
                    self._turn_started_at = asyncio.get_running_loop().time()
                LOGGER.info("Speech detected.")
                continue

            if event_type == "input_audio_buffer.speech_stopped":
                self._speech_active = False
                self._turn_started_at = None
                LOGGER.info("Speech ended. Waiting for transcription.")
                continue

            if event_type == "input_audio_buffer.committed":
                self._manual_commit_pending = False
                self._manual_commit_sent_at = None
                item_id = str(self._event_field(event, "item_id", "") or "")
                if not item_id:
                    continue
                previous_item_id = self._event_field(event, "previous_item_id")
                self._buffered_chunk_count = 0
                self._turn_started_at = (
                    asyncio.get_running_loop().time()
                    if self._speech_active or not self._uses_server_turn_detection()
                    else None
                )
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
                item_id = str(self._event_field(event, "item_id", "") or "")
                delta = str(self._event_field(event, "delta", "") or "")
                if not item_id or not delta:
                    continue
                next_text = f"{partial_text_by_item.get(item_id, '')}{delta}"
                partial_text_by_item[item_id] = next_text
                await transcript_queue.put(
                    TranscriptUpdate(item_id=item_id, text=next_text, is_partial=True)
                )
                continue

            if event_type == "conversation.item.input_audio_transcription.segment":
                item_id = str(self._event_field(event, "item_id", "") or "")
                segment_text = str(self._event_field(event, "text", "") or "").strip()
                if not item_id or not segment_text:
                    continue
                await transcript_queue.put(
                    TranscriptUpdate(item_id=item_id, text=segment_text, is_partial=True)
                )
                continue

            if event_type == "conversation.item.input_audio_transcription.completed":
                self._manual_commit_pending = False
                self._manual_commit_sent_at = None
                self._buffered_chunk_count = 0
                if not self._uses_server_turn_detection():
                    self._turn_started_at = asyncio.get_running_loop().time()
                item_id = str(self._event_field(event, "item_id", "") or "")
                transcript = str(self._event_field(event, "transcript", "") or "").strip()
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
                self._manual_commit_pending = False
                self._manual_commit_sent_at = None
                self._buffered_chunk_count = 0
                if not self._uses_server_turn_detection():
                    self._turn_started_at = asyncio.get_running_loop().time()
                item_id = str(self._event_field(event, "item_id", "") or "")
                error = self._event_field(event, "error")
                LOGGER.warning(
                    "Input audio transcription failed%s: %s",
                    f" for {item_id}" if item_id else "",
                    self._format_transcription_failure(error),
                )
                if item_id:
                    partial_text_by_item.pop(item_id, None)
                    completed_text_by_item.pop(item_id, None)
                    pending_commits.pop(item_id, None)
                    commit_index.pop(item_id, None)
                    emitted_items.add(item_id)
                    await self._emit_ready_completed(
                        transcript_queue,
                        pending_commits,
                        commit_index,
                        completed_text_by_item,
                        emitted_items,
                    )
                continue

            if event_type == "error":
                if self._is_benign_small_commit_error(event):
                    self._manual_commit_pending = False
                    self._manual_commit_sent_at = None
                    if self._speech_active or not self._uses_server_turn_detection():
                        self._turn_started_at = asyncio.get_running_loop().time()
                    LOGGER.debug(
                        "Ignoring undersized audio buffer commit while forcing a long turn."
                    )
                    continue
                raise UserFacingError(self._format_error_event(event))

    async def _force_commit_long_turns(self, connection: ClientConnection) -> None:
        while True:
            await asyncio.sleep(0.2)
            if self._manual_commit_pending:
                pending_seconds = self._manual_commit_pending_seconds()
                if pending_seconds < MANUAL_COMMIT_TIMEOUT_SECONDS:
                    continue
                self._manual_commit_pending = False
                self._manual_commit_sent_at = None
                self._turn_started_at = asyncio.get_running_loop().time()
                if not self._manual_commit_timeout_warning_emitted:
                    LOGGER.warning(
                        "Realtime transcription commit was pending for %.1fs. "
                        "Recovering so audio commits can continue. If this happens "
                        "with background music, use a mixer send with pastor mic only.",
                        pending_seconds,
                    )
                    self._manual_commit_timeout_warning_emitted = True
                continue
            if self._turn_started_at is None:
                self._turn_started_at = asyncio.get_running_loop().time()
                continue

            if self._uses_server_turn_detection() and not self._speech_active:
                continue

            elapsed_ms = (
                asyncio.get_running_loop().time() - self._turn_started_at
            ) * 1000
            if elapsed_ms < self._config.max_turn_ms:
                continue
            if self._buffered_audio_ms() < 250:
                continue

            self._manual_commit_pending = True
            self._manual_commit_sent_at = asyncio.get_running_loop().time()
            LOGGER.info(
                "Forcing audio commit after %.1fs of continuous speech.",
                elapsed_ms / 1000,
            )
            try:
                await self._send_event(connection, {"type": "input_audio_buffer.commit"})
            except Exception:
                self._manual_commit_pending = False
                self._manual_commit_sent_at = None
                raise

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
        error = self._event_field(event, "error")
        message = self._error_field(error, "message")
        if message:
            return f"OpenAI realtime error: {message}"
        return f"OpenAI realtime error: {event}"

    def _format_transcription_failure(self, error) -> str:
        message = self._error_field(error, "message")
        code = self._error_field(error, "code")
        error_type = self._error_field(error, "type")
        details = [str(value) for value in (message, code, error_type) if value]
        return " | ".join(details) if details else "Unknown transcription failure."

    def _buffered_audio_ms(self) -> int:
        return self._buffered_chunk_count * self._config.chunk_duration_ms

    def _manual_commit_pending_seconds(self) -> float:
        if self._manual_commit_sent_at is None:
            return 0.0
        return asyncio.get_running_loop().time() - self._manual_commit_sent_at

    def _uses_server_turn_detection(self) -> bool:
        return self._build_turn_detection() is not None

    def _is_benign_small_commit_error(self, event) -> bool:
        error = self._event_field(event, "error")
        if error is None:
            return False
        message = str(self._error_field(error, "message", "") or "").lower()
        code = str(self._error_field(error, "code", "") or "").lower()
        return (
            "input audio buffer is empty" in message
            or (
                "input audio buffer" in message
                and "empty" in message
            )
            or "buffer too small" in message
            or (
                "input audio buffer" in message
                and "too small" in message
            )
            or code == "input_audio_buffer_empty"
            or code == "input_audio_buffer_too_small"
        )

    def _event_field(self, event, name: str, default=None):
        if isinstance(event, dict):
            return event.get(name, default)
        return getattr(event, name, default)

    def _error_field(self, error, name: str, default=None):
        if isinstance(error, dict):
            return error.get(name, default)
        return getattr(error, name, default)
