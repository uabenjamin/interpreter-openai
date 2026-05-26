from __future__ import annotations

import asyncio
import logging

from .config import parse_args
from .error_handling import UserFacingError, classify_openai_error
from .instance_lock import InstanceLock, status_message, stop_running_instance
from .logging_utils import configure_logging
from .pipeline import InterpreterApp


LOGGER = logging.getLogger(__name__)


async def _run(config) -> int:
    app = InterpreterApp(config)
    if config.command == "doctor":
        await app.doctor()
        return 0
    if config.command == "devices":
        await app.list_devices()
        return 0
    if config.command == "status":
        LOGGER.info("%s", status_message())
        return 0
    if config.command == "stop":
        LOGGER.info("%s", stop_running_instance())
        return 0
    await app.run()
    return 0


def main() -> None:
    config = parse_args()
    configure_logging(
        level=logging.WARNING if config.command in {"run", "gui"} else logging.INFO
    )
    try:
        if config.command == "gui":
            from .gui import run_gui

            run_gui(config)
            raise SystemExit(0)
        if config.command == "run":
            with InstanceLock():
                raise SystemExit(asyncio.run(_run(config)))
        raise SystemExit(asyncio.run(_run(config)))
    except KeyboardInterrupt:
        LOGGER.info("Interrupted. Exiting.")
        raise SystemExit(130) from None
    except UserFacingError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from None
    except Exception as exc:
        classified = classify_openai_error(exc)
        if classified is not None:
            LOGGER.error("%s", classified)
            raise SystemExit(2) from None
        LOGGER.exception("Interpreter failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
