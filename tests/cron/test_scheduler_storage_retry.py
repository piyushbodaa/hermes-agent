"""run_job wiring for #138160's storage-busy bounded re-runs.

The classifier, ladder, and bookkeeping contracts live in
``tests/cron/test_unreachable_retry.py``; this file proves the production
``run_job`` exception tail hands the failed turn result to that classifier
and flags the in-flight job. Kept separate because ``test_scheduler.py`` is
over its FILE_LINES cap and may only shrink.
"""

from unittest.mock import MagicMock, patch

from cron import unreachable_retry as ur
from tests.cron.test_scheduler import _run_owned_job


def test_run_job_flags_storage_busy_result(tmp_path):
    job = {
        "id": "storage-busy-job", "name": "test", "prompt": "hello",
        "model": "test-model",
    }
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda session_id: session_id
    seen = {}
    real_flag = ur.flag_transient_retry_failure

    def capture_flag(job_arg, exc, agent=None, result=None):
        seen["result"] = result
        seen["cause"] = real_flag(job_arg, exc, agent, result)
        seen["flag"] = job_arg.get("_session_storage_busy")
        return seen["cause"]

    locked = {
        "failed": True,
        "completed": False,
        "failure_reason": "session_persistence_failed:locked",
        "error": "session storage was busy",
        "final_response": "",
        "current_turn_user_idx": 0,
        "messages": [{"role": "user", "content": "hello"}],
    }
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
             "api_key": "test-key", "base_url": "https://example.invalid/v1",
             "provider": "openrouter", "api_mode": "chat_completions"}), \
         patch("cron.unreachable_retry.flag_transient_retry_failure", capture_flag), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent._incremental_persistence_failed = False
        mock_agent.session_api_calls = 0
        mock_agent.run_conversation.return_value = locked
        mock_agent_cls.return_value = mock_agent
        success, _output, _final, error = _run_owned_job(job, tmp_path, fake_db)

    assert success is False
    assert "session storage was busy" in error
    assert seen["result"] == locked
    assert seen["cause"] == "storage_busy"
    assert seen["flag"] is True
