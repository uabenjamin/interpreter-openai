from __future__ import annotations

import asyncio
import html
import logging
import queue
import threading
from dataclasses import replace

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QCloseEvent, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - depends on local GUI install.
    PYSIDE_IMPORT_ERROR: ImportError | None = exc
    QApplication = None  # type: ignore[assignment]
    QCloseEvent = object  # type: ignore[assignment]
    QMainWindow = object  # type: ignore[assignment]
    QTextCursor = object  # type: ignore[assignment]
else:
    PYSIDE_IMPORT_ERROR = None

from .audio_io import (
    AudioUnavailableError,
    get_default_microphone_name,
    list_microphone_names,
)
from .config import AppConfig
from .error_handling import UserFacingError, classify_openai_error
from .instance_lock import InstanceLock
from .pipeline import InterpreterApp


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_LABEL = "System default input"


class GuiLogHandler(logging.Handler):
    def __init__(self, event_queue: queue.SimpleQueue[tuple[str, object]]) -> None:
        super().__init__(level=logging.WARNING)
        self._event_queue = event_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._event_queue.put(("log", self.format(record)))
        except Exception:
            self.handleError(record)


class InterpreterWindow(QMainWindow):  # type: ignore[misc]
    def __init__(self, config: AppConfig) -> None:
        if PYSIDE_IMPORT_ERROR is not None:
            raise UserFacingError(
                "The GUI requires PySide6. Install project dependencies with "
                "`.venv/bin/pip install -e .`, then run the GUI again."
            ) from PYSIDE_IMPORT_ERROR

        super().__init__()
        self._base_config = config
        self._events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._stopping = False

        self.setWindowTitle("Interpreter OpenAI")
        self.resize(1020, 720)
        self.setMinimumSize(760, 520)
        self._build_ui()
        self._load_input_devices()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain_events)
        self._timer.start(80)

        self._log_handler = GuiLogHandler(self._events)
        self._log_handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(self._log_handler)

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f1ea;
                color: #1f2933;
                font-family: "Avenir Next";
            }
            QLabel#title {
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#subtitle, QLabel#status {
                color: #52606d;
            }
            QComboBox {
                background: #fffaf0;
                border: 1px solid #c7bca1;
                border-radius: 6px;
                padding: 7px 10px;
                min-height: 24px;
            }
            QPushButton {
                background: #1f2933;
                color: #fffaf0;
                border: 0;
                border-radius: 8px;
                padding: 9px 18px;
                font-weight: 700;
            }
            QPushButton:disabled {
                background: #9aa5b1;
            }
            QPushButton#quitButton {
                background: #7c2d12;
            }
            QTextEdit {
                background: #fffaf0;
                border: 1px solid #e2d8bf;
                border-radius: 12px;
                padding: 12px 14px;
                font-size: 18px;
            }
            """
        )

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Live Interpreter")
        title.setObjectName("title")
        subtitle = QLabel(f"Target: {self._base_config.target_language_label}")
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(subtitle)
        layout.addLayout(header)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Input source"))
        self._input_combo = QComboBox()
        controls.addWidget(self._input_combo, stretch=1)

        self._start_stop_button = QPushButton("Start")
        self._start_stop_button.clicked.connect(self._toggle_running)
        controls.addWidget(self._start_stop_button)

        self._quit_button = QPushButton("Quit")
        self._quit_button.setObjectName("quitButton")
        self._quit_button.clicked.connect(self.close)
        controls.addWidget(self._quit_button)
        layout.addLayout(controls)

        self._status = QLabel("Select an input source, then click Start.")
        self._status.setObjectName("status")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        layout.addWidget(self._text, stretch=1)
        self.setCentralWidget(root)

    def _load_input_devices(self) -> None:
        try:
            devices = list_microphone_names()
        except AudioUnavailableError as exc:
            devices = []
            self._append_html(
                f'<p style="color:#b42318;"><b>Audio devices unavailable:</b> {html.escape(str(exc))}</p>'
            )

        self._input_combo.clear()
        self._input_combo.addItem(DEFAULT_INPUT_LABEL)
        for device in devices:
            self._input_combo.addItem(device)

        preferred = self._preferred_input_device(devices)
        index = self._input_combo.findText(preferred)
        self._input_combo.setCurrentIndex(index if index >= 0 else 0)

    def _preferred_input_device(self, devices: list[str]) -> str:
        if self._base_config.input_device:
            selector = self._base_config.input_device.casefold()
            for device in devices:
                if selector in device.casefold():
                    return device
        for device in devices:
            if "maono" in device.casefold():
                return device
        for device in devices:
            lowered = device.casefold()
            if "macbook" in lowered or "built-in" in lowered or "builtin" in lowered:
                return device
        try:
            default = get_default_microphone_name()
        except AudioUnavailableError:
            default = ""
        return default if default in devices else DEFAULT_INPUT_LABEL

    def _toggle_running(self) -> None:
        if self._running:
            self._stop_interpreter()
        else:
            self._start_interpreter()

    def _start_interpreter(self) -> None:
        if self._running:
            return
        selected_input = self._input_combo.currentText().strip()
        input_device = None if selected_input == DEFAULT_INPUT_LABEL else selected_input
        config = replace(
            self._base_config,
            command="run",
            input_device=input_device,
            input_audio_file=None,
        )
        self._running = True
        self._stopping = False
        self._start_stop_button.setText("Stop")
        self._input_combo.setEnabled(False)
        self._append_status("Starting interpreter...")
        self._set_status("Starting...")
        self._thread = threading.Thread(
            target=self._run_interpreter_thread,
            args=(config,),
            name="interpreter-gui-worker",
            daemon=True,
        )
        self._thread.start()

    def _stop_interpreter(self) -> None:
        if not self._running or self._stopping:
            return
        self._stopping = True
        self._set_status("Stopping...")
        self._start_stop_button.setEnabled(False)
        self._append_status("Stopping interpreter...")
        loop = self._loop
        task = self._task
        if loop is not None and task is not None:
            loop.call_soon_threadsafe(task.cancel)

    def _run_interpreter_thread(self, config: AppConfig) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        app = InterpreterApp(
            config,
            output_handler=self._on_pipeline_output,
            status_handler=self._on_pipeline_status,
        )
        try:
            with InstanceLock():
                self._task = loop.create_task(app.run())
                loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except UserFacingError as exc:
            self._events.put(("error", str(exc)))
        except BaseException as exc:
            classified = classify_openai_error(exc)
            self._events.put(("error", classified or f"Interpreter failed: {exc}"))
        finally:
            self._task = None
            self._loop = None
            loop.close()
            self._events.put(("stopped", None))

    def _on_pipeline_output(self, label: str, sequence_id: int | None, text: str) -> None:
        self._events.put(("output", (label, sequence_id, text)))

    def _on_pipeline_status(self, text: str) -> None:
        self._events.put(("status", text))

    def _drain_events(self) -> None:
        while True:
            try:
                event_type, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if event_type == "status":
                self._set_status(str(payload))
                self._append_status(str(payload))
            elif event_type == "output":
                label, sequence_id, text = payload
                self._append_output(str(label), int(sequence_id), str(text))
            elif event_type == "log":
                self._append_error(str(payload))
            elif event_type == "error":
                self._set_status(str(payload))
                self._append_error(str(payload))
            elif event_type == "stopped":
                self._running = False
                self._stopping = False
                self._start_stop_button.setText("Start")
                self._start_stop_button.setEnabled(True)
                self._input_combo.setEnabled(True)
                self._set_status("Stopped.")
                self._append_status("Interpreter stopped.")

    def _append_output(self, label: str, sequence_id: int, text: str) -> None:
        escaped = html.escape(text)
        if label == "target":
            self._append_html(
                f'<p style="margin:1px 0 9px; color:#0f172a; font-size:21px; '
                f'font-weight:650; line-height:1.22;">'
                f'<span style="color:#0f766e; font-size:11px; font-weight:700;">译 {sequence_id}</span>'
                f'&nbsp;&nbsp;{escaped}</p>'
            )
            return
        self._append_html(
            f'<p style="margin:7px 0 1px; color:#1f2933; font-size:19px; '
            f'line-height:1.2;">'
            f'<span style="color:#8a4b0f; font-size:11px; font-weight:700;">EN {sequence_id}</span>'
            f'&nbsp;&nbsp;{escaped}</p>'
        )

    def _append_status(self, text: str) -> None:
        self._append_html(
            f'<p style="margin:2px 0; color:#6b7280; font-size:12px;">[status] {html.escape(text)}</p>'
        )

    def _append_error(self, text: str) -> None:
        self._append_html(
            f'<p style="margin:3px 0; color:#b42318; font-size:13px; font-weight:700;">{html.escape(text)}</p>'
        )

    def _append_html(self, value: str) -> None:
        self._text.moveCursor(QTextCursor.MoveOperation.End)
        self._text.insertHtml(value)
        self._text.insertPlainText("\n")
        self._text.ensureCursorVisible()

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name.
        if self._running:
            answer = QMessageBox.question(
                self,
                "Quit Interpreter",
                "The interpreter is still running. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_interpreter()
            event.ignore()
            QTimer.singleShot(400, self._finish_close_when_stopped)
            return
        logging.getLogger().removeHandler(self._log_handler)
        event.accept()

    def _finish_close_when_stopped(self) -> None:
        if self._running:
            QTimer.singleShot(200, self._finish_close_when_stopped)
            return
        self.close()


def run_gui(config: AppConfig) -> None:
    if PYSIDE_IMPORT_ERROR is not None:
        raise UserFacingError(
            "The GUI requires PySide6. Install project dependencies with "
            "`.venv/bin/pip install -e .`, then run `.venv/bin/python -m interpreter_openai gui`."
        ) from PYSIDE_IMPORT_ERROR
    app = QApplication.instance() or QApplication([])
    window = InterpreterWindow(config)
    window.show()
    app.exec()
