"""simulation package.

Importing this package also silences gym's noisy import-time "upgrade to
Gymnasium" notice. gym/__init__.py prints a version-keyed message from
``gym_notices`` straight to stderr (it is a raw ``print``, not a
``warnings.warn``, so it cannot be filtered with the warnings module). Blanking
that mapping before ``import gym`` runs suppresses it. This runs here because
``simulation/__init__.py`` is imported before any ``simulation.*_sim`` module
body (which import the gym-based envs) — in both the main process and the
spawn-based evaluation subprocesses.
"""
try:  # pragma: no cover - best-effort suppression
    import gym_notices.notices as _gym_notices
    _gym_notices.notices = {}
except Exception:
    pass
