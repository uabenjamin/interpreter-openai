from __future__ import annotations

import os
import signal
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from .error_handling import UserFacingError


LOCK_PATH = Path.home() / "interpreter-openai" / "interpreter-openai.pid"


@dataclass(slots=True)
class RunningInstance:
    pid: int
    lock_path: Path


def _read_pid(lock_path: Path = LOCK_PATH) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def get_running_instance(lock_path: Path = LOCK_PATH) -> RunningInstance | None:
    pid = _read_pid(lock_path)
    if pid is None:
        return None
    if is_process_alive(pid):
        return RunningInstance(pid=pid, lock_path=lock_path)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    return None


class InstanceLock(AbstractContextManager["InstanceLock"]):
    def __init__(self, lock_path: Path = LOCK_PATH) -> None:
        self._lock_path = lock_path
        self._pid = os.getpid()
        self._held = False

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def acquire(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        running = get_running_instance(self._lock_path)
        if running is not None and running.pid != self._pid:
            raise UserFacingError(
                f"Interpreter is already running with PID {running.pid}. "
                "Use 'interpreter-openai stop' or send Control-C to that terminal first."
            )
        self._lock_path.write_text(f"{self._pid}\n", encoding="utf-8")
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        current_pid = _read_pid(self._lock_path)
        if current_pid == self._pid:
            try:
                self._lock_path.unlink()
            except FileNotFoundError:
                pass
        self._held = False


def stop_running_instance(lock_path: Path = LOCK_PATH) -> str:
    running = get_running_instance(lock_path)
    if running is None:
        return "No running interpreter-openai instance found."

    os.kill(running.pid, signal.SIGINT)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not is_process_alive(running.pid):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            return f"Stopped interpreter-openai PID {running.pid}."
        time.sleep(0.1)

    os.kill(running.pid, signal.SIGTERM)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not is_process_alive(running.pid):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            return f"Stopped interpreter-openai PID {running.pid}."
        time.sleep(0.1)

    raise UserFacingError(
        f"Interpreter-openai PID {running.pid} did not exit after SIGINT/SIGTERM."
    )


def status_message(lock_path: Path = LOCK_PATH) -> str:
    running = get_running_instance(lock_path)
    if running is None:
        return "Interpreter-openai is not running."
    return f"Interpreter-openai is running with PID {running.pid}."
