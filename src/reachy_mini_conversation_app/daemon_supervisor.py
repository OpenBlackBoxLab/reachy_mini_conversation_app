"""Auto-spawn and supervise a ``reachy-mini-daemon`` subprocess.

Used when the conversation app is launched with ``--auto-daemon``: if the
daemon isn't reachable on ``localhost:8000``, we start it ourselves and tear
it down when the app exits.
"""

from __future__ import annotations

import atexit
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
READY_TIMEOUT_SECS = 90
POLL_INTERVAL_SECS = 1.0


def _status_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/api/daemon/status"


def _fetch_status(host: str, port: int, timeout: float = 1.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(_status_url(host, port), timeout=timeout) as resp:
            if resp.status != 200:
                return None
            import json

            return json.loads(resp.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return None


def is_daemon_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Return True if the daemon HTTP server answers and reports state==running."""
    status = _fetch_status(host, port)
    return bool(status) and status.get("state") == "running"


def is_daemon_reachable(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Return True if anything answers on the daemon's status endpoint."""
    return _fetch_status(host, port) is not None


def _find_mjpython() -> Optional[pathlib.Path]:
    try:
        import mujoco
    except ImportError:
        return None
    candidate = pathlib.Path(mujoco.__file__).parent / "mjpython" / "mjpython"
    return candidate if candidate.exists() else None


def _find_daemon_entrypoint() -> Optional[pathlib.Path]:
    path = shutil.which("reachy-mini-daemon")
    return pathlib.Path(path) if path else None


def _build_command(viewer: bool, robot_name: Optional[str]) -> list[str]:
    daemon = _find_daemon_entrypoint()
    if daemon is None:
        raise RuntimeError(
            "Could not find the 'reachy-mini-daemon' script on PATH. "
            "Is the reachy_mini package installed in this environment?"
        )

    args: list[str] = ["--sim"]
    if not viewer:
        args.append("--headless")
    if robot_name is not None:
        args.extend(["--robot-name", robot_name])

    if viewer and sys.platform == "darwin":
        mjpy = _find_mjpython()
        if mjpy is None:
            raise RuntimeError(
                "Auto-daemon with viewer requires mjpython on macOS, but it "
                "could not be located next to the mujoco package. Either "
                "install mujoco, or drop --auto-daemon-viewer to run headless."
            )
        return [str(mjpy), str(daemon), *args]

    return [str(daemon), *args]


class DaemonSupervisor:
    """Spawns and reaps a reachy-mini-daemon subprocess."""

    def __init__(self, proc: subprocess.Popen, logger: logging.Logger) -> None:
        self._proc = proc
        self._logger = logger
        self._stopped = False
        atexit.register(self.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except ValueError:
                pass

    def _signal_handler(self, signum: int, frame) -> None:  # type: ignore[no-untyped-def]
        self.stop()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._proc.poll() is not None:
            return
        self._logger.info("Stopping auto-spawned reachy-mini-daemon (pid=%d)", self._proc.pid)
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._logger.warning("Daemon did not exit in 10s, killing")
                self._proc.kill()
                self._proc.wait(timeout=5)
        except Exception as e:
            self._logger.warning("Error while stopping daemon: %s", e)


def ensure_daemon(
    logger: logging.Logger,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    viewer: bool = False,
    robot_name: Optional[str] = None,
    ready_timeout: float = READY_TIMEOUT_SECS,
) -> Optional[DaemonSupervisor]:
    """Ensure the daemon is reachable, spawning one if needed.

    Returns the supervisor wrapping the spawned subprocess, or None if a daemon
    was already running. Raises RuntimeError if spawning fails or readiness
    times out.
    """
    if is_daemon_running(host, port):
        logger.info("Daemon already running on %s:%d, reusing it.", host, port)
        return None

    if is_daemon_reachable(host, port):
        logger.warning(
            "A process is responding on %s:%d but daemon state is not 'running'; "
            "will not spawn a second one. Stop or fix the existing daemon first.",
            host,
            port,
        )
        return None

    cmd = _build_command(viewer=viewer, robot_name=robot_name)
    logger.info("Auto-spawning daemon: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )

    supervisor = DaemonSupervisor(proc, logger)

    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            supervisor.stop()
            raise RuntimeError(
                f"Daemon subprocess exited prematurely with code {proc.returncode}. "
                "Check the daemon log above for the cause."
            )
        if is_daemon_running(host, port):
            logger.info("Daemon ready (pid=%d).", proc.pid)
            return supervisor
        time.sleep(POLL_INTERVAL_SECS)

    supervisor.stop()
    raise RuntimeError(
        f"Daemon did not reach 'running' state within {ready_timeout:.0f}s."
    )
