from argparse import Namespace
from unittest.mock import Mock

import numpy as np
import pytest

try:
    import torch

    from dvclive.fabric import DVCLiveLogger
except ImportError:
    pytest.skip("skipping lightning tests", allow_module_level=True)


class BoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


@pytest.mark.parametrize("step_idx", [10, None])
def test_dvclive_log_metrics(tmp_path, mocked_dvc_repo, step_idx):
    logger = DVCLiveLogger(dir=tmp_path)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": torch.tensor(0.1),
        "IntTensor": torch.tensor(1),
    }
    logger.log_metrics(metrics, step_idx)


def test_dvclive_log_hyperparams(tmp_path, mocked_dvc_repo):
    logger = DVCLiveLogger(dir=tmp_path)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar="buzz")),
        "layer": torch.nn.BatchNorm1d,
        "tensor": torch.empty(2, 2, 2),
        "array": np.empty([2, 2, 2]),
    }
    logger.log_hyperparams(hparams)


def test_dvclive_finalize(monkeypatch, tmp_path, mocked_dvc_repo):
    """Test that the SummaryWriter closes in finalize."""
    import dvclive

    monkeypatch.setattr(dvclive, "Live", Mock())
    logger = DVCLiveLogger(dir=tmp_path)
    assert logger._experiment is None
    logger.finalize("any")

    # no log calls, no experiment created -> nothing to flush
    logger.experiment.assert_not_called()

    logger = DVCLiveLogger(dir=tmp_path)
    logger.log_hyperparams({"flush_me": 11.1})  # trigger creation of an experiment
    logger.finalize("any")

    # finalize flushes to experiment directory
    logger.experiment.end.assert_called()
