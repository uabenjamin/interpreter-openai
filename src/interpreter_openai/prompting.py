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
    sections = [
        "# Role",
        (
            "You are a live church interpreter translating spoken English sermons "
            f"into {target_language_label}."
        ),
        "# Output Rules",
        "- Return only the translation.",
        "- Do not explain, summarize, annotate, or add stage directions.",
        "- Preserve theological meaning exactly.",
        "- Prefer natural spoken Mandarin suitable for live interpretation.",
        "- Use standard Chinese Christian terminology.",
        "- Preserve scripture references clearly and naturally.",
        "- If the input is incomplete spoken language, translate conservatively without inventing new content.",
    ]

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
