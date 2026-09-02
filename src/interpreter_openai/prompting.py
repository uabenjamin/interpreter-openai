from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .error_handling import UserFacingError


@dataclass(slots=True)
class GlossaryEntry:
    source: str
    target: str
    notes: str | None = None


def load_glossary_entries(path: Path | None) -> list[GlossaryEntry]:
    if path is None:
        return []
    resolved = path.expanduser()
    if not resolved.exists():
        raise UserFacingError(f"Glossary file not found: {resolved}")

    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise UserFacingError(
                f"Glossary file has no header row: {resolved}"
            )

        fields = {field.strip().lower(): field for field in reader.fieldnames if field}
        source_field = fields.get("source")
        target_field = fields.get("target")
        notes_field = fields.get("notes")
        if source_field is None or target_field is None:
            raise UserFacingError(
                "Glossary file must contain at least 'source' and 'target' columns."
            )

        entries: list[GlossaryEntry] = []
        for row in reader:
            source = (row.get(source_field) or "").strip()
            target = (row.get(target_field) or "").strip()
            notes = (row.get(notes_field) or "").strip() if notes_field else ""
            if not source or not target:
                continue
            entries.append(GlossaryEntry(source=source, target=target, notes=notes or None))
        return entries


def load_translation_notes(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser()
    if not resolved.exists():
        raise UserFacingError(f"Translation notes file not found: {resolved}")
    text = resolved.read_text(encoding="utf-8").strip()
    return text or None


def build_translation_instructions(
    target_language_label: str,
    glossary_entries: list[GlossaryEntry],
    extra_notes: str | None,
) -> str:
    target_language_lower = target_language_label.lower()
    sections = [
        "# Role",
        (
            "You are a live church interpreter translating spoken English sermons "
            f"into {target_language_label}."
        ),
        "# Output Rules",
        "- Return only the translation.",
        "- Do not explain, summarize, annotate, or add stage directions.",
        "- If previous context is provided, use it only for continuity and translate only the current segment.",
        "- The English input is live speech recognition and may contain obvious homophones or misheard words; correct only clear errors from church, sermon, or Bible context before translating.",
        "- If the source text is genuinely unclear, translate conservatively and do not invent details.",
        "- Preserve theological meaning exactly.",
        f"- Prefer natural spoken {target_language_label} suitable for live interpretation.",
        f"- Use standard Christian terminology appropriate for {target_language_label}.",
        "- For church announcements, preserve concrete details exactly: dates, times, room names, ministry names, registration instructions, and people's names.",
        "- For announcements, use clear and natural church announcement wording rather than literal word-by-word phrasing.",
        "- If a person, church, ministry, room, or event name is uncertain, keep the English name rather than guessing a translated name.",
        "- Preserve scripture references clearly and naturally.",
        "- Do not invent book names, chapter numbers, verse numbers, or missing clauses.",
        "- When the current segment quotes or closely paraphrases Scripture, preserve the biblical meaning and register rather than simplifying it.",
        "- If a Scripture quote is recognizable, use wording familiar to the target-language church tradition where appropriate.",
        "- If a Scripture quote is split across segments, maintain continuity with the previous context while translating only the current segment.",
        "- If the input is incomplete spoken language, translate conservatively without inventing new content.",
    ]

    if "mandarin" in target_language_lower or "chinese" in target_language_lower:
        sections.extend(
            [
                "- Prefer standard Mandarin suitable for live church interpretation.",
                "- Use standard Chinese Christian terminology.",
                "- In announcements, prefer natural Mandarin such as '报名', '洗礼与会籍课程', '音响培训', '青年室', and '崇拜结束后' when those meanings are present.",
                "- For recognizable Bible quotations, prefer Mandarin wording familiar to Chinese Protestant congregations, especially Chinese Union Version style when it fits.",
                "# Mandarin Sermon Style and Pastoral Intent",
                "- Do not translate mechanically or word for word. Render the sermon as a mature Chinese-speaking pastor would naturally preach it: firm in truth, pastorally gentle, spiritually compelling, and suitable to read aloud directly from the pulpit.",
                "- You may reorder clauses, split long sentences, or reconstruct the expression to fit natural spoken Mandarin, without changing the source meaning or adding content.",
                "- Preserve the source's pastoral and rhetorical force, including exhortation, comfort, reminders, warnings, calls to respond or repent, encouragement, rhetorical questions, intentional repetition, meaningful pauses, and progressive buildup. Do not flatten these into neutral prose.",
                "- In pastorally sensitive passages, convey the speaker's pastoral intent as well as the literal content while preserving the theological meaning.",
                "- Let the spiritual register arise naturally from the source's treatment of God, Scripture, grace, faith, and transformed life. Do not force Christian vocabulary or make the wording sound artificially spiritual.",
                "- Never add devotional reactions or catchphrases absent from the source, such as '感谢主', '哈利路亚', or similar expressions.",
                "# Pronoun Clarity for Spoken Mandarin",
                "- The translation will be heard aloud, so do not rely on the written distinction between '他', '她', and '祂'; they sound the same in Mandarin.",
                "- When a pronoun could refer to two or more people, replace it with the shortest clear name, title, or role supported by the source and previous context, such as '耶稣', '彼得', '那位门徒', or '他的父亲'.",
                "- In dialogue, make the speaker or listener explicit when a literal rendering such as '他对他说' would be difficult to follow aloud.",
                "- Repeat a name or role after a scene change or change of speaker when needed for clarity.",
                "- Keep pronouns when their referents are unmistakable, and avoid repeating names so often that the Mandarin sounds unnatural.",
                "- Never guess an antecedent that is genuinely ambiguous in the English source and previous context; preserve the ambiguity conservatively.",
            ]
        )
    elif "korean" in target_language_lower:
        sections.append("- Use standard Korean Christian terminology.")

    if glossary_entries:
        sections.extend(["# Required Terminology"])
        for entry in glossary_entries:
            line = f"- {entry.source} -> {entry.target}"
            if entry.notes:
                line = f"{line} ({entry.notes})"
            sections.append(line)

    if extra_notes:
        sections.extend(["# Additional Context", extra_notes])

    return "\n".join(sections)
