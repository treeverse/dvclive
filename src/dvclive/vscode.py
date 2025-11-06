import json
import os
from typing import Optional, Union

from dvclive.dvc import _find_dvc_root
from dvclive.utils import StrPath

from . import env


def _dvc_exps_run_dir(dirname: StrPath) -> str:
    return os.path.join(dirname, ".dvc", "tmp", "exps", "run")


def _dvclive_only_signal_file(root_dir: StrPath) -> str:
    dvc_exps_run_dir = _dvc_exps_run_dir(root_dir)
    return os.path.join(dvc_exps_run_dir, "DVCLIVE_ONLY")


def _dvclive_step_completed_signal_file(root_dir: StrPath) -> str:
    dvc_exps_run_dir = _dvc_exps_run_dir(root_dir)
    return os.path.join(dvc_exps_run_dir, "DVCLIVE_STEP_COMPLETED")


def _find_non_queue_root() -> Optional[str]:
    return os.getenv(env.DVC_ROOT) or _find_dvc_root()


def _write_file(file: str, contents: dict[str, Union[str, int]]):
    import builtins

    with builtins.open(file, "w", encoding="utf-8") as fobj:
        # NOTE: force flushing/writing empty file to disk, otherwise when
        # run in certain contexts (pytest) file may not actually be written
        fobj.write(json.dumps(contents, sort_keys=True, ensure_ascii=False))
        fobj.flush()
        os.fsync(fobj.fileno())


def mark_dvclive_step_completed(step: int) -> None:
    """
    https://github.com/iterative/vscode-dvc/issues/4528
    Signal DVC VS Code extension that
    a step has been completed for an experiment running in the queue
    """
    non_queue_root_dir = _find_non_queue_root()

    if not non_queue_root_dir:
        return

    exp_run_dir = _dvc_exps_run_dir(non_queue_root_dir)
    os.makedirs(exp_run_dir, exist_ok=True)

    signal_file = _dvclive_step_completed_signal_file(non_queue_root_dir)

    _write_file(signal_file, {"pid": os.getpid(), "step": step})


def cleanup_dvclive_step_completed() -> None:
    non_queue_root_dir = _find_non_queue_root()

    if not non_queue_root_dir:
        return

    signal_file = _dvclive_step_completed_signal_file(non_queue_root_dir)

    if not os.path.exists(signal_file):
        return

    os.remove(signal_file)


def mark_dvclive_only_started(exp_name: str) -> None:
    """
    Signal DVC VS Code extension that
    an experiment is running in the workspace.
    """
    root_dir = _find_dvc_root()
    if not root_dir:
        return

    exp_run_dir = _dvc_exps_run_dir(root_dir)
    os.makedirs(exp_run_dir, exist_ok=True)

    signal_file = _dvclive_only_signal_file(root_dir)

    _write_file(signal_file, {"pid": os.getpid(), "exp_name": exp_name})


def mark_dvclive_only_ended() -> None:
    root_dir = _find_dvc_root()
    if not root_dir:
        return

    signal_file = _dvclive_only_signal_file(root_dir)

    if not os.path.exists(signal_file):
        return

    os.remove(signal_file)
