# mypy: disable-error-code="no-redef"
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Union

from typing_extensions import override

try:
    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    from lightning.pytorch.loggers.logger import Logger
    from lightning.pytorch.loggers.utilities import _scan_checkpoints
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    from pytorch_lightning.callbacks.model_checkpoint import (  # type: ignore[assignment]
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers.logger import Logger  # type: ignore[assignment]
    from pytorch_lightning.utilities import rank_zero_only

    try:
        from pytorch_lightning.utilities.logger import _scan_checkpoints
    except ImportError:
        from pytorch_lightning.loggers.utilities import (  # type: ignore[assignment]
            _scan_checkpoints,
        )


from dvclive.fabric import DVCLiveLogger as FabricDVCLiveLogger


def _should_sync():
    """
    Find out if pytorch_lightning is calling `log_metrics` from the functions
    where we actually want to sync.
    For example, prevents calling sync when external callbacks call
    `log_metrics` or during the multiple `update_eval_step_metrics`.
    """
    return any(
        frame.function
        in (
            "update_train_step_metrics",
            "update_train_epoch_metrics",
            "log_eval_end_metrics",
        )
        for frame in inspect.stack()
    )


class DVCLiveLogger(Logger, FabricDVCLiveLogger):
    def __init__(
        self,
        run_name: Optional[str] = "dvclive_run",
        prefix="",
        log_model: Union[str, bool] = False,
        experiment=None,
        **kwargs,
    ):
        super().__init__(
            run_name=run_name,
            prefix=prefix,
            experiment=experiment,
            **kwargs,
        )
        self._log_model = log_model
        self._logged_model_time: dict[str, float] = {}
        self._checkpoint_callback: Optional[ModelCheckpoint] = None
        self._all_checkpoint_paths: list[str] = []

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Mapping[str, Union[int, float, str]],
        step: Optional[int] = None,
        sync: Optional[bool] = False,
    ) -> None:
        if not sync and _should_sync():
            sync = True
        super().log_metrics(metrics, step, sync)

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if self._log_model in [True, "all"]:
            self._checkpoint_callback = checkpoint_callback
            self._scan_checkpoints(checkpoint_callback)
        if self._log_model == "all" or (
            self._log_model is True and checkpoint_callback.save_top_k == -1
        ):
            self._save_checkpoints(checkpoint_callback)

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Log best model.
        if self._checkpoint_callback:
            self._scan_checkpoints(self._checkpoint_callback)
            self._save_checkpoints(self._checkpoint_callback)
            best_model_path = self._checkpoint_callback.best_model_path
            self.experiment.log_artifact(
                best_model_path, name="best", type="model", copy=True
            )
        super().finalize(status)

    def _scan_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # update model time and append path to list of all checkpoints
        for t, p, _, _ in checkpoints:
            self._logged_model_time[p] = t
            self._all_checkpoint_paths.append(p)

    def _save_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # drop unused checkpoints
        if not self.experiment._resume and checkpoint_callback.dirpath:  # noqa: SLF001
            for p in Path(checkpoint_callback.dirpath).iterdir():
                if str(p) not in self._all_checkpoint_paths:
                    p.unlink(missing_ok=True)

        # save directory
        self.experiment.log_artifact(checkpoint_callback.dirpath)
