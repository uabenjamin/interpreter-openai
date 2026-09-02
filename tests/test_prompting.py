from __future__ import annotations

import unittest

from interpreter_openai.prompting import build_translation_instructions


class TranslationInstructionTests(unittest.TestCase):
    def test_mandarin_prompt_includes_pastoral_sermon_style(self) -> None:
        instructions = build_translation_instructions(
            "Mandarin Chinese (Simplified Chinese script)",
            glossary_entries=[],
            extra_notes=None,
        )

        self.assertIn("# Mandarin Sermon Style and Pastoral Intent", instructions)
        self.assertIn("Do not translate mechanically or word for word", instructions)
        self.assertIn("firm in truth, pastorally gentle", instructions)
        self.assertIn("reorder clauses, split long sentences", instructions)
        self.assertIn("pastoral and rhetorical force", instructions)
        self.assertIn("calls to respond or repent", instructions)
        self.assertIn("Do not flatten these into neutral prose", instructions)
        self.assertIn("speaker's pastoral intent", instructions)
        self.assertIn("Do not force Christian vocabulary", instructions)
        self.assertIn("Never add devotional reactions", instructions)
        self.assertIn("感谢主", instructions)
        self.assertIn("哈利路亚", instructions)

    def test_mandarin_specific_style_is_not_added_for_other_languages(self) -> None:
        instructions = build_translation_instructions(
            "Korean",
            glossary_entries=[],
            extra_notes=None,
        )

        self.assertNotIn("# Mandarin Sermon Style and Pastoral Intent", instructions)
        self.assertNotIn("Do not translate mechanically or word for word", instructions)


if __name__ == "__main__":
    unittest.main()
