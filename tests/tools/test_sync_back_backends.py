"""Tests for backend-specific bulk download implementations and cleanup() wiring."""

import asyncio
import io
import sys
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.environments import ssh as ssh_env
from tools.environments import modal as modal_env
from tools.environments import daytona as daytona_env
from tools.environments.ssh import SSHEnvironment

# ── SSH helpers ──────────────────────────────────────────────────────

@pytest.fixture
def ssh_mock_env(monkeypatch):
    """Create an SSHEnvironment with mocked connection/sync."""
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        ssh_env, "FileSyncManager",
        lambda **kw: type("M", (), {
            "sync": lambda self, **k: None,
            "sync_back": lambda self: None,
        })(),
    )
    return SSHEnvironment(host="example.com", user="testuser")

# ── Modal helpers ────────────────────────────────────────────────────

def _make_mock_modal_env():
    """Create a minimal ModalEnvironment without calling __init__."""
    env = object.__new__(modal_env.ModalEnvironment)
    env._sandbox = MagicMock()
    env._worker = MagicMock()
    env._persistent = False
    env._task_id = "test"
    env._sync_manager = None
    return env

def _wire_modal_download(env, *, tar_bytes=b"fake-tar-data", exit_code=0):
    """Wire sandbox.exec.aio to return mock tar output for download tests.

    Returns the exec_calls list for assertion.
    """
    exec_calls = []

    async def mock_exec_fn(*args, **kwargs):
        exec_calls.append(args)
        proc = MagicMock()
        proc.stdout = MagicMock()
        assert kwargs["text"] is False
        proc.stdout.__aiter__.return_value = [tar_bytes]
        proc.stdout.aclose = AsyncMock()
        proc.wait = MagicMock()
        proc.wait.aio = AsyncMock(return_value=exit_code)
        return proc

    env._sandbox.exec = MagicMock()
    env._sandbox.exec.aio = mock_exec_fn

    def real_run_coroutine(coro, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    env._worker.run_coroutine = real_run_coroutine
    return exec_calls

# ── Daytona helpers ──────────────────────────────────────────────────

def _make_mock_daytona_env():
    """Create a minimal DaytonaEnvironment without calling __init__."""
    env = object.__new__(daytona_env.DaytonaEnvironment)
    env._sandbox = MagicMock()
    env._sandbox.process.exec.return_value.exit_code = 0
    response = env._sandbox.fs._api_client.download_file_without_preload_content.return_value
    response.status = 200
    response.stream.return_value = [b"archive"]
    env._remote_home = "/root"
    env._sync_manager = None
    env._lock = __import__("threading").Lock()
    env._persistent = True
    env._task_id = "test"
    env._daytona = MagicMock()
    return env

# =====================================================================
# SSH bulk download
# =====================================================================

def _mock_ssh_tar(completed):
    process = MagicMock()
    process.stdout = io.BytesIO(completed.stdout or b"")
    process.stderr = io.BytesIO(completed.stderr or b"")
    process.wait.return_value = completed.returncode
    process.__enter__.return_value = process
    return patch.object(subprocess, "Popen", return_value=process)


class TestSSHBulkDownload:
    """Unit tests for _ssh_bulk_download."""

    def test_ssh_bulk_download_runs_tar_over_ssh(self, ssh_mock_env, tmp_path):
        """subprocess.run command should include tar cf - over SSH."""
        dest = tmp_path / "backup.tar"

        with _mock_ssh_tar(subprocess.CompletedProcess([], 0)) as mock_run:
            ssh_mock_env._ssh_bulk_download(dest)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "tar cf -" in cmd_str
        assert "-C /" in cmd_str
        assert "home/testuser/.hermes" in cmd_str
        assert "ssh" in cmd_str
        assert "testuser@example.com" in cmd_str

    @pytest.mark.parametrize(("limit", "accepted"), [(4, False), (5, True)])
    def test_ssh_limits_actual_transfer(self, ssh_mock_env, tmp_path, monkeypatch, limit, accepted):
        from tools.environments.file_sync import SyncBackSizeLimitExceeded

        monkeypatch.setattr("tools.environments.file_sync._sync_back_max_bytes", lambda: limit)
        monkeypatch.setattr(ssh_mock_env, "_build_ssh_command",
                            lambda: [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'hello')"])
        destination = tmp_path / "out"
        if accepted:
            ssh_mock_env._ssh_bulk_download(destination)
            assert destination.read_bytes() == b"hello"
        else:
            with pytest.raises(SyncBackSizeLimitExceeded):
                ssh_mock_env._ssh_bulk_download(destination)
            assert destination.stat().st_size <= limit

    def test_ssh_bulk_download_tolerates_only_socket_ignored_exit_2(self, ssh_mock_env, tmp_path):
        """Live sockets are excluded up front, and an rc=2 whose stderr is solely
        'socket ignored' lines (a socket not named *.sock) does not fail the transfer."""
        dest = tmp_path / "backup.tar"
        stderr = b"tar: home/testuser/.hermes/gateway.sock: socket ignored\n"
        completed = subprocess.CompletedProcess([], 2, stderr=stderr)

        with _mock_ssh_tar(completed) as mock_run:
            ssh_mock_env._ssh_bulk_download(dest)  # must not raise

        assert "--exclude='*.sock'" in " ".join(mock_run.call_args[0][0])

    def test_ssh_bulk_download_still_fails_on_every_other_status(self, ssh_mock_env, tmp_path):
        """rc=1; rc=2 with a real error beside the socket line, with no diagnostic at all, or
        with 'socket ignored' merely inside a filename — all still raise."""
        from tools.environments.base import EnvironmentConnectionError
        dest = tmp_path / "backup.tar"
        failures = (
            subprocess.CompletedProcess([], 1, stderr=b"tar: home/testuser/.hermes/state.db: file changed as we read it"),
            subprocess.CompletedProcess([], 2, stderr=(b"tar: home/testuser/.hermes/gateway.sock: socket ignored\n"
                                                      b"tar: home/testuser/.hermes/state.db: Cannot open: Permission denied\n")),
            subprocess.CompletedProcess([], 2, stderr=b"\n"),
            subprocess.CompletedProcess([], 2, stderr=b"tar: socket ignored dir/state.db: Cannot open: Permission denied\n"),
        )
        for completed in failures:
            with _mock_ssh_tar(completed):
                with pytest.raises(EnvironmentConnectionError):
                    ssh_mock_env._ssh_bulk_download(dest)

class TestSSHCleanup:
    """Verify SSH cleanup() calls sync_back() before closing ControlMaster."""

    def test_ssh_cleanup_calls_sync_back_before_control_exit(self, monkeypatch):
        """sync_back() must run before the ControlMaster exit command."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)

        call_order = []

        class TrackingSyncManager:
            def __init__(self, **kwargs):
                pass

            def sync(self, **kw):
                pass

            def sync_back(self):
                call_order.append("sync_back")

        monkeypatch.setattr(ssh_env, "FileSyncManager", TrackingSyncManager)

        env = SSHEnvironment(host="h", user="u")

        # Create a fake control socket so cleanup tries the SSH exit
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sock") as tmp:
            env.control_socket = Path(tmp.name)

        def mock_run(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "-O" in cmd and "exit" in cmd_str:
                call_order.append("control_exit")
            return subprocess.CompletedProcess([], 0)

        with patch.object(subprocess, "run", side_effect=mock_run):
            env.cleanup()

        assert call_order.index("sync_back") < call_order.index("control_exit")

# =====================================================================
# Modal bulk download
# =====================================================================

class TestModalBulkDownload:
    """Unit tests for _modal_bulk_download."""

    def test_modal_bulk_download_command(self, tmp_path):
        """exec should be called with tar cf - -C /root/.hermes ."""
        env = _make_mock_modal_env()
        exec_calls = _wire_modal_download(env, tar_bytes=b"tar-content")
        dest = tmp_path / "backup.tar"

        env._modal_bulk_download(dest)

        assert dest.read_bytes() == b"tar-content"
        assert len(exec_calls) == 1
        args = exec_calls[0]
        assert args[0] == "bash"
        assert args[1] == "-c"
        assert "tar cf -" in args[2]
        assert "-C / root/.hermes" in args[2]
        # Live sockets cannot be archived; exclude them like the SSH backend.
        assert "--exclude='*.sock'" in args[2]

class TestModalCleanup:
    """Verify Modal cleanup() calls sync_back() before terminate."""

    def test_modal_cleanup_calls_sync_back(self):
        """cleanup() should call sync_back() before sandbox.terminate."""
        env = _make_mock_modal_env()

        call_order = []
        sync_mgr = MagicMock()
        sync_mgr.sync_back = lambda: call_order.append("sync_back")
        env._sync_manager = sync_mgr

        # Mock terminate to track call order
        async def mock_terminate():
            pass

        env._sandbox.terminate = MagicMock()
        env._sandbox.terminate.aio = mock_terminate
        env._worker.run_coroutine = lambda coro, **kw: (
            call_order.append("terminate"),
            asyncio.new_event_loop().run_until_complete(coro),
        )
        env._worker.stop = lambda: None

        env.cleanup()

        assert "sync_back" in call_order
        assert call_order.index("sync_back") < call_order.index("terminate")

# =====================================================================
# Daytona bulk download
# =====================================================================

class TestDaytonaBulkDownload:
    """Unit tests for _daytona_bulk_download."""

    def test_daytona_bulk_download_creates_tar_and_downloads(self, tmp_path):
        """exec and download_file should both be called."""
        env = _make_mock_daytona_env()
        dest = tmp_path / "backup.tar"

        env._daytona_bulk_download(dest)

        # exec called twice: tar creation + rm cleanup
        assert env._sandbox.process.exec.call_count == 2
        tar_cmd = env._sandbox.process.exec.call_args_list[0][0][0]
        assert "tar cf" in tar_cmd
        # Live sockets cannot be archived; exclude them like the SSH backend.
        assert "--exclude='*.sock'" in tar_cmd
        # PID-suffixed temp path avoids collisions on sync_back retry
        assert "/tmp/.hermes_sync." in tar_cmd
        assert ".tar" in tar_cmd
        assert ".hermes" in tar_cmd

        cleanup_cmd = env._sandbox.process.exec.call_args_list[1][0][0]
        assert "rm -f" in cleanup_cmd
        assert "/tmp/.hermes_sync." in cleanup_cmd

        download = env._sandbox.fs._api_client.download_file_without_preload_content
        download.assert_called_once()
        assert download.call_args.kwargs["path"].startswith("/tmp/.hermes_sync.")
        assert download.call_args.kwargs["path"].endswith(".tar")
        assert dest.read_bytes() == b"archive"
        download.return_value.close.assert_called_once()
        download.return_value.release_conn.assert_called_once()

    def test_daytona_bulk_download_uses_remote_home(self, tmp_path):
        """The tar command should use the env's _remote_home."""
        env = _make_mock_daytona_env()
        env._remote_home = "/home/daytona"
        dest = tmp_path / "backup.tar"

        env._daytona_bulk_download(dest)

        tar_cmd = env._sandbox.process.exec.call_args_list[0][0][0]
        assert "home/daytona/.hermes" in tar_cmd

@pytest.mark.parametrize("backend", ["daytona", "modal"])
def test_sdk_transfer_checks_real_bytes(tmp_path, monkeypatch, backend):
    from tools.environments.file_sync import SyncBackSizeLimitExceeded

    monkeypatch.setattr("tools.environments.file_sync._sync_back_max_bytes", lambda: 3)
    if backend == "modal":
        env = _make_mock_modal_env()
        _wire_modal_download(env, tar_bytes=b"hello")
        download = env._modal_bulk_download
    else:
        env = _make_mock_daytona_env()
        download = env._daytona_bulk_download
    destination = tmp_path / "out"
    with pytest.raises(SyncBackSizeLimitExceeded):
        download(destination)
    assert destination.stat().st_size <= 3


class TestDaytonaCleanup:
    """Verify Daytona cleanup() calls sync_back() before stop."""

    def test_daytona_cleanup_calls_sync_back(self):
        """cleanup() should call sync_back() before sandbox.stop()."""
        env = _make_mock_daytona_env()

        call_order = []
        sync_mgr = MagicMock()
        sync_mgr.sync_back = lambda: call_order.append("sync_back")
        env._sync_manager = sync_mgr
        env._sandbox.stop = lambda: call_order.append("stop")

        env.cleanup()

        assert "sync_back" in call_order
        assert "stop" in call_order
        assert call_order.index("sync_back") < call_order.index("stop")

# =====================================================================
# FileSyncManager wiring: bulk_download_fn passed by each backend
# =====================================================================
