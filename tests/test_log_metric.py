import math
import os

import numpy as np
import pytest

from dvclive import Live
from dvclive.error import DvcLiveError, InvalidDataTypeError
from dvclive.plots import Metric
from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics, parse_tsv


def test_logging_no_step(tmp_dir):
    dvclive = Live("logs")

    dvclive.log_metric("m1", 1, plot=False)
    dvclive.make_summary()

    assert not (tmp_dir / "logs" / "plots" / "metrics" / "m1.tsv").is_file()
    assert (tmp_dir / dvclive.metrics_file).is_file()

    s = load_yaml(dvclive.metrics_file)
    assert s["m1"] == 1
    assert "step" not in s


@pytest.mark.parametrize("path", ["logs", os.path.join("subdir", "logs")])
def test_logging_step(tmp_dir, path):
    dvclive = Live(path)
    dvclive.log_metric("m1", 1)
    dvclive.next_step()
    assert (tmp_dir / dvclive.dir).is_dir()
    assert (tmp_dir / dvclive.plots_dir / Metric.subfolder / "m1.tsv").is_file()
    assert (tmp_dir / dvclive.metrics_file).is_file()

    s = load_yaml(dvclive.metrics_file)
    assert s["m1"] == 1
    assert s["step"] == 0


def test_nested_logging(tmp_dir):
    dvclive = Live("logs")

    out = tmp_dir / dvclive.plots_dir / Metric.subfolder

    dvclive.log_metric("train/m1", 1)
    dvclive.log_metric("val/val_1/m1", 1)
    dvclive.log_metric("val/val_1/m2", 1)

    dvclive.next_step()

    assert (out / "val" / "val_1").is_dir()
    assert (out / "train" / "m1.tsv").is_file()
    assert (out / "val" / "val_1" / "m1.tsv").is_file()
    assert (out / "val" / "val_1" / "m2.tsv").is_file()

    assert "m1" in parse_tsv(out / "train" / "m1.tsv")[0]
    assert "m1" in parse_tsv(out / "val" / "val_1" / "m1.tsv")[0]
    assert "m2" in parse_tsv(out / "val" / "val_1" / "m2.tsv")[0]

    summary = load_yaml(dvclive.metrics_file)

    assert summary["train"]["m1"] == 1
    assert summary["val"]["val_1"]["m1"] == 1
    assert summary["val"]["val_1"]["m2"] == 1


@pytest.mark.parametrize(
    ("parent", "name"), [("epoch", "epoch/loss"), ("train/epoch", "train/epoch/loss")]
)
@pytest.mark.parametrize("val", [0, 1.0, "value"])
@pytest.mark.parametrize("plot", [True, False])
def test_log_metric_parent_conflict(tmp_dir, parent, name, val, plot):
    live = Live("logs")
    live.log_metric(parent, val)
    summary = live.summary.copy()
    history = {path: path.read_bytes() for path in tmp_dir.rglob("*.tsv")}

    with pytest.raises(
        DvcLiveError,
        match=f"Cannot log metric '{name}': parent '{parent}' already has a value",
    ):
        live.log_metric(name, 0.5, plot=plot)

    assert live.summary == summary
    assert name not in live._metrics
    assert {path: path.read_bytes() for path in tmp_dir.rglob("*.tsv")} == history


def test_log_metric_resumed_parent_conflict(tmp_dir):
    live = Live("logs")
    live.log_metric("epoch", 0, plot=False)
    live.make_summary()
    resumed = Live("logs", resume=True)

    with pytest.raises(DvcLiveError, match="parent 'epoch' already has a value"):
        resumed.log_metric("epoch/loss", 0.5)

    assert resumed.summary["epoch"] == 0
    assert not list(tmp_dir.rglob("*.tsv"))


def test_log_metric_step_parent_conflict(tmp_dir):
    live = Live("logs")
    live.next_step()

    with pytest.raises(DvcLiveError, match="parent 'step' already has a value"):
        live.log_metric("step/loss", 0.5)

    assert live.summary["step"] == 0
    assert not list(tmp_dir.rglob("*.tsv"))


def test_log_metric_nested_updates(tmp_dir):
    live = Live("logs")
    live.log_metric("train/loss", 1)
    live.log_metric("train/accuracy", 0.5)
    live.log_metric("train/loss", 0.25)
    live.log_metric("validation/loss", 0.75)
    live.make_summary()

    assert load_yaml(live.metrics_file) == {
        "train": {"loss": 0.25, "accuracy": 0.5},
        "validation": {"loss": 0.75},
    }


@pytest.mark.parametrize("timestamp", [True, False])
def test_log_metric_timestamp(tmp_dir, timestamp):
    live = Live()
    live.log_metric("foo", 1.0, timestamp=timestamp)
    live.next_step()

    history, _ = parse_metrics(live)
    logged = next(iter(history.values()))
    assert ("timestamp" in logged[0]) == timestamp


@pytest.mark.parametrize("invalid_type", [{0: 1}, [0, 1], (0, 1)])
def test_invalid_metric_type(tmp_dir, invalid_type):
    dvclive = Live()

    with pytest.raises(
        InvalidDataTypeError,
        match=f"Data 'm' has not supported type {type(invalid_type)}",
    ):
        dvclive.log_metric("m", invalid_type)


@pytest.mark.parametrize(
    ("val"),
    [math.inf, math.nan, np.nan, np.inf],
)
def test_log_metric_inf_nan(tmp_dir, val):
    with Live() as live:
        live.log_metric("metric", val)
    assert live.summary["metric"] == str(val)


def test_log_metic_str(tmp_dir):
    with Live() as live:
        live.log_metric("metric", "foo")
    assert live.summary["metric"] == "foo"
