"""Custom exceptions for the workflow system.

Replaces bare `sys.exit(1)` calls with typed exceptions so callers can:
- Catch and handle specific failure modes
- Allow cleanup (wandb.finish, tmux kill) via try/finally
- Distinguish between different error categories
- Avoid aborting Hydra multirun sweeps on single-trial failure
"""


class WorkflowError(Exception):
    """Base error for workflow pipeline failures.

    Raised when a workflow operation cannot proceed due to:
    - Missing or invalid config
    - W&B API failures (sweep/run not found)
    - Invalid parameter combinations
    - Session management failures

    The @task_wrapper decorator catches this in its except clause,
    ensuring wandb.finish() and other cleanup run in the finally block.
    """


class ConfigError(WorkflowError):
    """Invalid or missing configuration."""


class SweepError(WorkflowError):
    """W&B sweep operation failed (creation, lookup, or agent)."""


class EvaluationError(WorkflowError):
    """Evaluation task failed (no runs, timeout, aggregation)."""


class SessionError(WorkflowError):
    """tmux session management failure."""
