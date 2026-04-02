from __future__ import annotations

import asyncio
import logging
import os

from openai import OpenAI

from .config import AppConfig
from .error_handling import UserFacingError


LOGGER = logging.getLogger(__name__)


def build_client(config: AppConfig) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise UserFacingError(
            "OPENAI_API_KEY is not set. Save it in your shell profile, for example "
            "~/.zshrc, and restart the shell."
        )
    return OpenAI(api_key=api_key, project=config.openai_project)


async def verify_openai_text_generation(client: OpenAI, model: str) -> str:
    def _run() -> str:
        response = client.responses.create(
            model=model,
            instructions="Reply with only the word OK.",
            input="Confirm API access.",
            max_output_tokens=16,
        )
        return (response.output_text or "").strip()

    return await asyncio.to_thread(_run)
