from __future__ import annotations

import asyncio
from collections import deque

from openai import OpenAI

from .prompting import build_translation_instructions, load_glossary_entries, load_translation_notes


MAX_CONTEXT_CHARS = 1400


class OpenAITranslator:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_output_tokens: int,
        target_language_label: str,
        glossary_file,
        translation_notes_file,
    ) -> None:
        glossary_entries = load_glossary_entries(glossary_file)
        translation_notes = load_translation_notes(translation_notes_file)
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._target_language_label = target_language_label
        self._recent_context: deque[tuple[str, str]] = deque(maxlen=6)
        self._instructions = build_translation_instructions(
            target_language_label,
            glossary_entries,
            translation_notes,
        )

    async def translate_text(self, english_text: str) -> str:
        return await asyncio.to_thread(self._translate_text_blocking, english_text)

    def _translate_text_blocking(self, english_text: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            instructions=self._instructions,
            input=self._build_translation_input(english_text),
            max_output_tokens=self._max_output_tokens,
            temperature=0.2,
        )
        translated = (response.output_text or "").strip()
        if not translated:
            raise RuntimeError("OpenAI translation returned empty output.")
        self._remember_context(english_text, translated)
        return translated

    def _build_translation_input(self, english_text: str) -> str:
        context = self._format_recent_context()
        if not context:
            return f"CURRENT SEGMENT TO TRANSLATE:\n{english_text}"
        return (
            "PREVIOUS SERMON CONTEXT FOR CONTINUITY ONLY. DO NOT RETRANSLATE IT:\n"
            f"{context}\n\n"
            "CURRENT SEGMENT TO TRANSLATE:\n"
            f"{english_text}"
        )

    def _format_recent_context(self) -> str:
        if not self._recent_context:
            return ""

        lines: list[str] = []
        remaining_chars = MAX_CONTEXT_CHARS
        for english, translated in reversed(self._recent_context):
            item = (
                f"- English: {english}\n"
                f"  {self._target_language_label}: {translated}"
            )
            item_len = len(item)
            if item_len > remaining_chars:
                break
            lines.append(item)
            remaining_chars -= item_len
        return "\n".join(reversed(lines))

    def _remember_context(self, english_text: str, translated_text: str) -> None:
        english = " ".join(english_text.split()).strip()
        translated = " ".join(translated_text.split()).strip()
        if english and translated:
            self._recent_context.append((english, translated))
