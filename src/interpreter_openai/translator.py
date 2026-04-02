from __future__ import annotations

import asyncio

from openai import OpenAI

from .prompting import build_translation_instructions, load_glossary_entries, load_translation_notes


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
            input=english_text,
            max_output_tokens=self._max_output_tokens,
            temperature=0.2,
        )
        translated = (response.output_text or "").strip()
        if not translated:
            raise RuntimeError("OpenAI translation returned empty output.")
        return translated
