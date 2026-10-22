"""Small terminal progress reporter for long Traj-LTLf runs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
import sys
import threading
import time
import unicodedata
from typing import Iterator, TextIO


ProgressMode = str


@dataclass(frozen=True)
class ProgressConfig:
    """Display configuration for terminal progress."""

    mode: ProgressMode = "auto"
    spinner_interval_seconds: float = 0.12


class TerminalProgress:
    """Print concise stage updates and optional TTY spinner lines."""

    def __init__(
        self,
        config: ProgressConfig | None = None,
        *,
        stream: TextIO | None = None,
    ) -> None:
        self.config = config or ProgressConfig()
        self.stream = stream or sys.stdout
        self.started_at = time.monotonic()
        self._lock = threading.Lock()
        self._spinner_stop: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None
        self._spinner_last_len = 0
        self._stage_index = 0
        self._mode = self._resolve_mode(self.config.mode)

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    def start_run(self, title: str, details: list[str] | None = None) -> None:
        if not self.enabled:
            return
        self._write_raw(title)
        for detail in details or []:
            self._write_raw(detail)
        self._write_raw("")

    def stage(
        self,
        title: str,
        *,
        step: int | None = None,
        total: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._stop_spinner()
        if step is None:
            self._stage_index += 1
            label = f"[{self._stage_index}]"
        elif total is None:
            label = f"[{step}]"
        else:
            label = f"[{step}/{total}]"
        self._line("RUN", f"{label} {title}")

    def info(self, message: str) -> None:
        self._line("INFO", message)

    def ok(self, message: str) -> None:
        self._line("OK", message)

    def warn(self, message: str) -> None:
        self._line("WARN", message)

    def wait(self, message: str, details: list[str] | None = None) -> None:
        if not self.enabled:
            return
        self._stop_spinner()
        self._line("WAIT", message)
        for detail in details or []:
            self._write_raw(f"       {detail}")

    def error(self, message: str) -> None:
        self._line("ERR", message)

    def done(self, message: str = "Traj-LTLf run finished") -> None:
        self._line("DONE", message)

    @contextmanager
    def task(self, message: str, *, waiting: bool = False) -> Iterator[None]:
        self.task_start(message, waiting=waiting)
        try:
            yield
        finally:
            self.task_stop()

    def task_start(self, message: str, *, waiting: bool = False) -> None:
        if not self.enabled:
            return
        self._stop_spinner()
        status = "WAIT" if waiting else "RUN"
        if self._mode == "dynamic":
            stop_event = threading.Event()
            self._spinner_stop = stop_event
            # Show the first frame before the blocking operation starts.
            self._render_spinner(message, "|", status)
            self._spinner_thread = threading.Thread(
                target=self._spin,
                args=(message, stop_event, status),
                daemon=True,
            )
            self._spinner_thread.start()
        elif waiting:
            self._write_raw("...")
        else:
            self._line(status, message)

    def task_stop(self) -> None:
        self._stop_spinner()

    def close(self) -> None:
        self._stop_spinner()

    def _resolve_mode(self, requested: ProgressMode) -> ProgressMode:
        normalized = requested.lower()
        if normalized in {"off", "none", "quiet"}:
            return "off"
        if normalized in {"plain", "text"}:
            return "plain"
        if normalized in {"dynamic", "spinner"}:
            return "dynamic" if self._is_tty() else "plain"
        return "dynamic" if self._is_tty() else "plain"

    def _is_tty(self) -> bool:
        isatty = getattr(self.stream, "isatty", None)
        return bool(isatty and isatty())

    def _line(self, status: str, message: str) -> None:
        if not self.enabled:
            return
        self._stop_spinner()
        self._write_raw(f"[{self._elapsed_text()}] [{status}] {message}")

    def _spin(
        self, message: str, stop_event: threading.Event,
        status: str,
    ) -> None:
        frames = ("|", "/", "-", "\\")
        index = 1
        while not stop_event.wait(self.config.spinner_interval_seconds):
            self._render_spinner(message, frames[index % len(frames)], status)
            index += 1

    def _render_spinner(
        self, message: str, frame: str, status: str,
    ) -> None:
        text = frame if status == "WAIT" else f"[{self._elapsed_text()}] [{status}] {frame} {message}"
        with self._lock:
            self.stream.write("\r" + text)
            self.stream.flush()
            # Chinese labels occupy two terminal columns per wide character.
            self._spinner_last_len = sum(
                0 if unicodedata.combining(char) else
                2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
                for char in text
            )

    def _stop_spinner(self) -> None:
        stop_event = self._spinner_stop
        thread = self._spinner_thread
        self._spinner_stop = None
        self._spinner_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=1.0)
        if self._mode == "dynamic" and self._spinner_last_len:
            with self._lock:
                self.stream.write("\r" + " " * self._spinner_last_len + "\r")
                self.stream.flush()
            self._spinner_last_len = 0

    def _write_raw(self, text: str) -> None:
        with self._lock:
            self.stream.write(text + "\n")
            self.stream.flush()

    def _elapsed_text(self) -> str:
        elapsed = int(time.monotonic() - self.started_at)
        return str(timedelta(seconds=elapsed))


@dataclass
class _ProgressSession:
    reporters: list[TerminalProgress]


_current_session: ContextVar[_ProgressSession | None] = ContextVar(
    "terminal_progress_session", default=None,
)


@contextmanager
def progress_session() -> Iterator[None]:
    """Scope CLI progress to this invocation and clean up even on Ctrl+C."""

    session = _ProgressSession(reporters=[])
    token = _current_session.set(session)
    try:
        yield
    finally:
        try:
            for reporter in reversed(session.reporters):
                reporter.close()
        finally:
            _current_session.reset(token)


@contextmanager
def model_wait() -> Iterator[None]:
    """Show API waiting in every CLI path, without changing library output."""

    session = _current_session.get()
    progress = session.reporters[-1] if session and session.reporters else None
    if progress is None:
        yield
        return
    with progress.task("", waiting=True):
        yield


def create_progress(mode: ProgressMode = "auto") -> TerminalProgress:
    """Create a terminal progress reporter."""

    progress = TerminalProgress(ProgressConfig(mode=mode))
    session = _current_session.get()
    if session is not None:
        session.reporters.append(progress)
    return progress
