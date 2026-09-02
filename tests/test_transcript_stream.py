from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from interpreter_openai.transcript_stream import OpenAIRealtimeTranscriber


class TranscriptOrderingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.transcriber = OpenAIRealtimeTranscriber.__new__(OpenAIRealtimeTranscriber)
        self.queue: asyncio.Queue = asyncio.Queue()

    async def test_empty_transcript_unblocks_the_following_turn(self) -> None:
        pending_commits = {"empty": None, "spoken": "empty"}
        commit_index = {"empty": 1, "spoken": 2}
        completed_text_by_item = {"spoken": "The sermon continues."}
        emitted_items: set[str] = set()

        await self.transcriber._emit_ready_completed(
            self.queue,
            pending_commits,
            commit_index,
            completed_text_by_item,
            emitted_items,
        )
        self.assertTrue(self.queue.empty())

        completed_text_by_item["empty"] = ""
        await self.transcriber._emit_ready_completed(
            self.queue,
            pending_commits,
            commit_index,
            completed_text_by_item,
            emitted_items,
        )

        update = self.queue.get_nowait()
        self.assertEqual(update.item_id, "spoken")
        self.assertEqual(update.text, "The sermon continues.")
        self.assertFalse(update.is_partial)
        self.assertTrue(self.queue.empty())
        self.assertEqual(emitted_items, {"empty", "spoken"})
        self.assertEqual(pending_commits, {})
        self.assertEqual(commit_index, {})

    async def test_later_completion_cannot_overtake_earlier_turn(self) -> None:
        pending_commits = {"first": None, "second": "first"}
        commit_index = {"first": 1, "second": 2}
        completed_text_by_item = {"second": "Second"}
        emitted_items: set[str] = set()

        await self.transcriber._emit_ready_completed(
            self.queue,
            pending_commits,
            commit_index,
            completed_text_by_item,
            emitted_items,
        )
        self.assertTrue(self.queue.empty())

        completed_text_by_item["first"] = "First"
        await self.transcriber._emit_ready_completed(
            self.queue,
            pending_commits,
            commit_index,
            completed_text_by_item,
            emitted_items,
        )

        self.assertEqual(self.queue.get_nowait().text, "First")
        self.assertEqual(self.queue.get_nowait().text, "Second")

    async def test_completion_waits_until_its_commit_is_observed(self) -> None:
        pending_commits: dict[str, str | None] = {}
        commit_index: dict[str, int] = {}
        completed_text_by_item = {"early": "Ready early"}
        emitted_items: set[str] = set()

        await self.transcriber._emit_ready_completed(
            self.queue,
            pending_commits,
            commit_index,
            completed_text_by_item,
            emitted_items,
        )
        self.assertTrue(self.queue.empty())

        pending_commits["early"] = None
        commit_index["early"] = 1
        await self.transcriber._emit_ready_completed(
            self.queue,
            pending_commits,
            commit_index,
            completed_text_by_item,
            emitted_items,
        )

        self.assertEqual(self.queue.get_nowait().text, "Ready early")


class TranscriptEventTests(unittest.IsolatedAsyncioTestCase):
    def make_transcriber(self, events: list[dict[str, object]]) -> OpenAIRealtimeTranscriber:
        config = SimpleNamespace(
            turn_detection_type="none",
            transcription_model="gpt-realtime-whisper",
        )
        transcriber = OpenAIRealtimeTranscriber(config, "test-key")

        async def iter_events(_connection):
            for event in events:
                yield event

        transcriber._iter_events = iter_events
        return transcriber

    async def test_empty_completed_event_does_not_block_later_text(self) -> None:
        transcriber = self.make_transcriber(
            [
                {"type": "transcription_session.updated"},
                {
                    "type": "input_audio_buffer.committed",
                    "item_id": "empty",
                    "previous_item_id": None,
                },
                {
                    "type": "input_audio_buffer.committed",
                    "item_id": "spoken",
                    "previous_item_id": "empty",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "spoken",
                    "transcript": "The next speaker begins.",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "empty",
                    "transcript": "",
                },
            ]
        )
        queue: asyncio.Queue = asyncio.Queue()
        ready = asyncio.Event()

        await transcriber._receive_events(object(), queue, ready)

        self.assertTrue(ready.is_set())
        update = queue.get_nowait()
        self.assertEqual(update.item_id, "spoken")
        self.assertEqual(update.text, "The next speaker begins.")
        self.assertTrue(queue.empty())

    async def test_failed_event_does_not_block_later_text(self) -> None:
        transcriber = self.make_transcriber(
            [
                {
                    "type": "input_audio_buffer.committed",
                    "item_id": "failed",
                    "previous_item_id": None,
                },
                {
                    "type": "input_audio_buffer.committed",
                    "item_id": "spoken",
                    "previous_item_id": "failed",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "spoken",
                    "transcript": "Still running.",
                },
                {
                    "type": "conversation.item.input_audio_transcription.failed",
                    "item_id": "failed",
                    "error": {"message": "No speech recognized"},
                },
            ]
        )
        queue: asyncio.Queue = asyncio.Queue()

        await transcriber._receive_events(object(), queue, asyncio.Event())

        update = queue.get_nowait()
        self.assertEqual(update.item_id, "spoken")
        self.assertEqual(update.text, "Still running.")
        self.assertTrue(queue.empty())


if __name__ == "__main__":
    unittest.main()
