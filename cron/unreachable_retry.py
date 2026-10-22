"""Automatic bounded re-runs for scheduled fires that provably ran no tools.

Inspired by Claude Cowork (desktop changelog v1.46388.1, 2026-09-04): "automatic re-runs
(after 5, 15, and 30 minutes) for a scheduled task that could not reach the model at all,
for example right after the computer wakes behind a VPN."

The classes are deliberately narrow:

* model-unreachable: a transient network/DNS failure with zero completed API calls;
* session-storage-busy: the turn failed closed on a locked session store before any
  tool call or tool result in the current turn (#138160).

Either way, re-running cannot double a tool side effect — unlike a generic failure
retry (see PR #16512), which has to answer for one-shot dispatch accounting and mid-run
side effects. Recurring jobs only: finite one-shots are pre-claimed by
``claim_dispatch`` (at-most-times, #38758) and must not regain a consumed dispatch here.

While a retry is pending the failure notice is suppressed (Cowork re-runs silently); a
run that gets past the transient gate — success or not — resets the ladder. Disable with
``cron.retry_unreachable: false`` in config.yaml.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from cron.constants import is_recurring
from hermes_time import now as _hermes_now

logger = logging.getLogger("cron.scheduler")

# Cowork's ladder: re-run after 5, 15, then 30 minutes; then give up until the
# schedule's own next occurrence.
RETRY_DELAYS_SECONDS: tuple[int, ...] = (300, 900, 1800)

MODEL_UNREACHABLE_CAUSE = "model_unreachable"
STORAGE_BUSY_CAUSE = "storage_busy"

# A locked session store has already outwaited the store's own write patience by
# the time the turn fails closed, so its ladder is shorter: two re-runs, then the
# schedule's own next occurrence.
STORAGE_BUSY_RETRY_DELAYS_SECONDS: tuple[int, ...] = (60, 300)

_RETRY_DELAYS_BY_CAUSE: dict[str, tuple[int, ...]] = {
    MODEL_UNREACHABLE_CAUSE: RETRY_DELAYS_SECONDS,
    STORAGE_BUSY_CAUSE: STORAGE_BUSY_RETRY_DELAYS_SECONDS,
}

# Persisted on the job while a retry cycle is active: {"attempt": <1-based count of
# retries already scheduled>, "at": <ISO instant of the pending retry>, "expr": <the cron
# expression it was planned under>, "cause": <which transient class parked it>}. Cleared
# by any run that gets past the transient gate.
STATE_KEY = "unreachable_retry"


def retry_enabled(cfg: Optional[dict] = None) -> bool:
    """``cron.retry_unreachable`` — default ON (spend-neutral: only fires when zero
    model calls were made)."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
        except Exception:  # config unreadable — keep the reliability default
            return True
    cron_cfg = (cfg or {}).get("cron")
    if not isinstance(cron_cfg, dict):
        return True
    return cron_cfg.get("retry_unreachable") is not False


def is_model_unreachable_failure(exc: BaseException, agent: Any = None) -> bool:
    """True when *exc* is a transient network/DNS failure and *agent* (may be ``None``)
    never completed a model call — the run consumed nothing and executed nothing."""
    if int(getattr(agent, "session_api_calls", 0) or 0) > 0:
        return False
    from cron.scheduler_preflight import _is_transient_provider_resolve_error

    return _is_transient_provider_resolve_error(exc)


def is_session_storage_busy_failure(result: Any, agent: Any = None) -> bool:
    """True for a locked-store turn failure proven to precede every tool effect.

    The finalizer stamps ``session_persistence_failed:locked`` only for the
    SQLite-lock cause bucket. That alone is not enough: the same failure can
    surface while persisting a completed tool round. Retry only when the
    current turn contains neither an assistant tool call nor a tool result,
    and incremental tool-progress persistence did not already fail. When the
    turn boundary is not proven (``current_turn_user_idx``), scan the whole
    transcript rather than risk repeating an action."""
    if not isinstance(result, dict) or result.get("failed") is not True:
        return False
    if str(result.get("failure_reason") or "") != "session_persistence_failed:locked":
        return False
    if getattr(agent, "_incremental_persistence_failed", False):
        return False
    messages = result.get("messages")
    if not isinstance(messages, (list, tuple)):
        return False
    start = result.get("current_turn_user_idx")
    turn_messages = (
        messages[start:]
        if isinstance(start, int) and not isinstance(start, bool) and 0 <= start < len(messages)
        else messages
    )
    return not any(
        isinstance(row, dict)
        and (row.get("role") == "tool" or (row.get("role") == "assistant" and row.get("tool_calls")))
        for row in turn_messages
    )


def flag_transient_retry_failure(
    job: dict[str, Any], exc: BaseException, agent: Any = None, result: Any = None,
) -> Optional[str]:
    """Flag a failed run for the bounded ladder; returns its cause or ``None``."""
    if is_model_unreachable_failure(exc, agent):
        job["_model_unreachable"] = True
        return MODEL_UNREACHABLE_CAUSE
    if is_session_storage_busy_failure(result, agent):
        job["_session_storage_busy"] = True
        return STORAGE_BUSY_CAUSE
    return None


def _normalize_cause(cause: Any) -> str:
    return cause if cause in _RETRY_DELAYS_BY_CAUSE else MODEL_UNREACHABLE_CAUSE


def _retry_cause(job: dict[str, Any]) -> str:
    if job.get("_session_storage_busy"):
        return STORAGE_BUSY_CAUSE
    if job.get("_model_unreachable"):
        return MODEL_UNREACHABLE_CAUSE
    return _normalize_cause((job.get(STATE_KEY) or {}).get("cause"))


def _retry_delays(cause: Any) -> tuple[int, ...]:
    return _RETRY_DELAYS_BY_CAUSE[_normalize_cause(cause)]


def _ladder_applies(job: dict[str, Any]) -> bool:
    """The ladder's applicability gate: recurring, not paused, and enabled in config."""
    return is_recurring(job) and job.get("state") != "paused" and retry_enabled()


def _ladder_instant(job: dict[str, Any], natural_next: Optional[datetime],
                    now: datetime, cause: Any = None) -> Optional[datetime]:
    """The ladder instant ``plan_retry`` parks for this flagged failure, or None when it
    parks nothing: not recurring, paused, disabled, ladder exhausted, or the schedule's
    own *natural_next* fires at or before the rung. The single source of that decision
    for ``plan_retry`` and ``will_retry``."""
    delays = _retry_delays(_retry_cause(job) if cause is None else cause)
    attempt = int((job.get(STATE_KEY) or {}).get("attempt") or 0)
    if attempt >= len(delays) or not _ladder_applies(job):
        return None
    # late: jobs imports this module's helpers
    from cron.jobs import _instant_at_or_before, _seconds_after

    retry_dt = _seconds_after(now, delays[attempt])
    if natural_next is not None and _instant_at_or_before(natural_next, retry_dt):
        return None  # the natural occurrence IS the retry
    return retry_dt


def will_retry(job: dict[str, Any]) -> bool:
    """True iff ``plan_retry`` will park a re-run for this flagged failure, so the scheduler
    may hold the interim notice. Called before ``mark_job_run``."""
    repeat = job.get("repeat") or {}
    times = repeat.get("times")
    counted = 0 if is_retry_run(job) else 1  # a ladder re-run does not count (``ladder_rung``)
    if times is not None and times > 0 and int(repeat.get("completed") or 0) + counted >= times:
        return False  # _advance_after_run completes the job; plan_retry never runs
    from cron.jobs import _parse_aware, compute_next_run

    now = _hermes_now()
    natural_next = _parse_aware(compute_next_run(job.get("schedule") or {}, now.isoformat()))
    if natural_next is None:
        # The natural occurrence is uncomputable (e.g. croniter missing): _advance_after_run
        # leaves the record state=error — terminal — so plan_retry never runs either.
        return False
    return _ladder_instant(job, natural_next, now) is not None


def clear_state(job: dict[str, Any]) -> None:
    """A run got past the transient gate (any outcome): the ladder resets."""
    job.pop(STATE_KEY, None)


def is_retry_fire(job: dict[str, Any], next_run: str) -> bool:
    """True for the exact ladder instant parked by ``plan_retry`` (off the cron lattice).

    The expression fingerprint keeps a direct ``jobs.json`` schedule edit from inheriting the
    exception, as in ``cron.quota_hold.is_recovery_fire``.
    """
    state = job.get(STATE_KEY) or {}
    return state.get("at") == next_run and state.get("expr") == (job.get("schedule") or {}).get("expr")


def is_retry_run(job: dict[str, Any]) -> bool:
    """True when this dispatch snapshot fired the ladder instant parked by ``plan_retry``: a
    re-run of an occurrence that already counted toward ``repeat``, so it must not count again.
    Manual runs carry no ``_scheduled_instant`` and always count."""
    from cron.occurrences import scheduled_instant

    instant = job.get("_scheduled_instant")
    return instant is not None and scheduled_instant((job.get(STATE_KEY) or {}).get("at")) == instant


def plan_retry(job: dict[str, Any], *, cause: Any = MODEL_UNREACHABLE_CAUSE) -> bool:
    """Called under the jobs lock AFTER ``_advance_after_run`` computed the schedule's
    natural ``next_run_at`` for a failed, flagged run. Pulls ``next_run_at`` earlier to
    the ladder instant when that is sooner than the natural occurrence; exhausted or
    inapplicable cycles clear state and leave the schedule untouched. Returns True when
    a retry was scheduled."""
    from cron.jobs import _parse_aware  # late: jobs imports this module's helpers

    cause = _normalize_cause(cause)
    delays = _retry_delays(cause)
    retry_dt = _ladder_instant(job, _parse_aware(job.get("next_run_at")), _hermes_now(), cause)
    attempt = int((job.get(STATE_KEY) or {}).get("attempt") or 0)
    if retry_dt is None:
        if attempt >= len(delays) and _ladder_applies(job):
            # Ladder exhausted: fall back to the natural schedule and reset so the NEXT
            # occurrence gets a fresh ladder if the transient condition persists.
            logger.warning(
                "Job '%s': %s after %d automatic re-runs — waiting for the "
                "scheduled occurrence at %s",
                job.get("name", job.get("id", "?")),
                "session storage still busy" if cause == STORAGE_BUSY_CAUSE else "model unreachable",
                attempt, job.get("next_run_at"))
        clear_state(job)
        return False
    delay = delays[attempt]
    retry_at = retry_dt.isoformat()
    job[STATE_KEY] = {
        "attempt": attempt + 1, "at": retry_at,
        "expr": job["schedule"].get("expr"), "cause": cause,
    }
    job["next_run_at"] = retry_at
    if job.get("state") != "paused":
        job["state"] = "scheduled"
    if cause == STORAGE_BUSY_CAUSE:
        logger.info(
            "Job '%s': session storage busy before any tool ran — automatic re-run %d/%d "
            "in %ds (at %s)",
            job.get("name", job.get("id", "?")), attempt + 1, len(delays), delay, retry_at)
    else:
        logger.info(
            "Job '%s': model unreachable with zero API calls — automatic re-run %d/%d in %ds "
            "(at %s)",
            job.get("name", job.get("id", "?")), attempt + 1, len(delays), delay, retry_at)
    return True
