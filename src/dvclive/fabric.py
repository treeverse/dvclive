# mypy: disable-error-code="no-redef"
from argparse import Namespace
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union

try:
    from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
    from lightning.fabric.utilities.logger import (
        _add_prefix,
        _convert_params,
        _sanitize_callable_params,
    )
    from lightning.fabric.utilities.rank_zero import rank_zero_only
except ImportError:
    from lightning_fabric.loggers.logger import (  # type: ignore[assignment]
        Logger,
        rank_zero_experiment,
    )
    from lightning_fabric.utilities.logger import (
        _add_prefix,
        _convert_params,
        _sanitize_callable_params,
    )
    from lightning_fabric.utilities.rank_zero import rank_zero_only

from torch import is_tensor

from dvclive.plots import Metric
from dvclive.utils import standardize_metric_name

if TYPE_CHECKING:
    from dvclive import Live


class DVCLiveLogger(Logger):
    LOGGER_JOIN_CHAR = "/"

    def __init__(
        self,
        run_name: Optional[str] = None,
        prefix: str = "",
        experiment: Optional["Live"] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._version = run_name
        self._prefix = prefix
        self._experiment = experiment
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "DvcLiveLogger"

    @property
    def version(self) -> Union[int, str]:
        if self._version is None:
            self._version = ""
        return self._version

    @property
    @rank_zero_experiment
    def experiment(self) -> "Live":
        if self._experiment is not None:
            return self._experiment

        assert (  # noqa: S101
            rank_zero_only.rank == 0  # type: ignore[attr-defined]
        ), "tried to init DVCLive in non global_rank=0"  # type: ignore[attr-defined]

        from dvclive import Live

        self._experiment = Live(**self._kwargs)

        return self._experiment

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Mapping[str, Union[int, float, str]],
        step: Optional[int] = None,
        sync: Optional[bool] = True,
    ) -> None:
        assert (  # noqa: S101
            rank_zero_only.rank == 0  # type: ignore[attr-defined]
        ), "experiment tried to log from global_rank != 0"

        if step:
            self.experiment.step = step
        else:
            self.experiment.step = self.experiment.step + 1

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)  # type: ignore[assignment,arg-type]

        for metric_name, metric_val in metrics.items():
            val = metric_val
            if is_tensor(val):  # type: ignore[unreachable]
                val = val.cpu().detach().item()  # type: ignore[union-attr,unreachable]
            name = standardize_metric_name(metric_name, __name__)
            if Metric.could_log(val):
                self.experiment.log_metric(name=name, val=val)
            else:
                raise ValueError(  # noqa: TRY003
                    f"\n you tried to log {val} which is currently not supported."
                    "Try a scalar/tensor."
                )

        if sync:
            self.experiment.sync()

    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        """Record hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
        """
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        params = self._sanitize_params(params)
        self.experiment.log_params(params)

    @rank_zero_only
    def finalize(self, status: str) -> None:  # noqa: ARG002
        if self._experiment is not None:
            self.experiment.end()

    @staticmethod
    def _sanitize_params(params: Union[dict[str, Any], Namespace]) -> dict[str, Any]:
        from argparse import Namespace

        # logging of arrays with dimension > 1 is not supported, sanitize as string
        params = {
            k: str(v) if hasattr(v, "ndim") and v.ndim > 1 else v
            for k, v in params.items()
        }

        # logging of argparse.Namespace is not supported, sanitize as string
        params = {
            k: str(v) if isinstance(v, Namespace) else v for k, v in params.items()
        }

        return params  # noqa: RET504

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state
