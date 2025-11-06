import os
from pathlib import Path

import pytest
from PIL import Image

from dvclive import Live
from dvclive.dvc import make_dvcyaml
from dvclive.error import InvalidDvcyamlError
from dvclive.serialize import dump_yaml, load_yaml


def test_make_dvcyaml_empty(tmp_dir):
    live = Live(dvcyaml="dvc.yaml")
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {}


def test_make_dvcyaml_param(tmp_dir):
    live = Live(dvcyaml="dvc.yaml")
    live.log_param("foo", 1)
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "params": ["dvclive/params.yaml"],
    }


def test_make_dvcyaml_metrics(tmp_dir):
    live = Live(dvcyaml="dvc.yaml")
    live.log_metric("bar", 2)
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["dvclive/metrics.json"],
        "plots": [{"dvclive/plots/metrics": {"x": "step"}}],
    }


def test_make_dvcyaml_metrics_no_plots(tmp_dir):
    live = Live(dvcyaml="dvc.yaml")
    live.log_metric("bar", 2, plot=False)
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["dvclive/metrics.json"],
    }


def test_make_dvcyaml_summary(tmp_dir):
    live = Live(dvcyaml="dvc.yaml")
    live.summary["bar"] = 2
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["dvclive/metrics.json"],
    }


def test_make_dvcyaml_all_plots(tmp_dir):
    live = Live(dvcyaml="dvc.yaml")
    live.log_param("foo", 1)
    live.log_metric("bar", 2)
    live.log_image("img.png", Image.new("RGB", (10, 10), (250, 250, 250)))
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [0, 1, 1, 0])
    live.log_sklearn_plot(
        "confusion_matrix",
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        name="confusion_matrix_normalized",
        normalized=True,
    )
    live.log_sklearn_plot("roc", [0, 0, 1, 1], [0.0, 0.5, 0.5, 0.0], "custom_name_roc")
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["dvclive/metrics.json"],
        "params": ["dvclive/params.yaml"],
        "plots": [
            {"dvclive/plots/metrics": {"x": "step"}},
            "dvclive/plots/images",
            {
                "dvclive/plots/sklearn/confusion_matrix.json": {
                    "template": "confusion",
                    "x": "actual",
                    "y": "predicted",
                    "title": "Confusion Matrix",
                    "x_label": "True Label",
                    "y_label": "Predicted Label",
                },
            },
            {
                "dvclive/plots/sklearn/confusion_matrix_normalized.json": {
                    "template": "confusion_normalized",
                    "title": "Confusion Matrix",
                    "x": "actual",
                    "x_label": "True Label",
                    "y": "predicted",
                    "y_label": "Predicted Label",
                }
            },
            {
                "dvclive/plots/sklearn/custom_name_roc.json": {
                    "template": "simple",
                    "x": "fpr",
                    "y": "tpr",
                    "title": "Receiver operating characteristic (ROC)",
                    "x_label": "False Positive Rate",
                    "y_label": "True Positive Rate",
                }
            },
        ],
    }


def test_make_dvcyaml_relpath(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()
    live = Live(dvcyaml="dir/dvc.yaml")
    live.log_metric("foo", 1)
    live.log_artifact("model.pth", type="model")
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["../dvclive/metrics.json"],
        "plots": [{"../dvclive/plots/metrics": {"x": "step"}}],
        "artifacts": {
            "model": {"path": "../model.pth", "type": "model"},
        },
    }


@pytest.mark.parametrize(
    ("orig_yaml", "updated_yaml"),
    [
        pytest.param(
            {"stages": {"train": {"cmd": "train.py"}}},
            {
                "stages": {"train": {"cmd": "train.py"}},
                "metrics": ["dvclive/metrics.json"],
                "plots": [
                    {"dvclive/plots/metrics": {"x": "step"}},
                ],
            },
            id="stages",
        ),
        pytest.param(
            {"params": ["dvclive/params.yaml"]},
            {
                "metrics": ["dvclive/metrics.json"],
                "plots": [{"dvclive/plots/metrics": {"x": "step"}}],
            },
            id="drop_extra_sections",
        ),
        pytest.param(
            {"plots": ["dvclive/plots/images"]},
            {
                "metrics": ["dvclive/metrics.json"],
                "plots": [{"dvclive/plots/metrics": {"x": "step"}}],
            },
            id="drop_unlogged_plots",
        ),
        pytest.param(
            {"plots": [{"dvclive/plots/metrics": {"x": "step", "y": "foo"}}]},
            {
                "metrics": ["dvclive/metrics.json"],
                "plots": [{"dvclive/plots/metrics": {"x": "step"}}],
            },
            id="plot_props",
        ),
        pytest.param(
            {
                "plots": [
                    {
                        "custom": {
                            "x": "step",
                            "y": {"dvclive/plots/metrics": "foo"},
                            "title": "custom",
                        }
                    },
                ],
            },
            {
                "metrics": ["dvclive/metrics.json"],
                "plots": [
                    {
                        "custom": {
                            "x": "step",
                            "y": {"dvclive/plots/metrics": "foo"},
                            "title": "custom",
                        }
                    },
                    {"dvclive/plots/metrics": {"x": "step"}},
                ],
            },
            id="keep_custom_plots",
        ),
    ],
)
def test_make_dvcyaml_update(tmp_dir, orig_yaml, updated_yaml):
    dump_yaml(orig_yaml, "dvc.yaml")

    live = Live(dvcyaml="dvc.yaml")
    live.log_metric("foo", 2)
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == updated_yaml


@pytest.mark.parametrize(
    ("orig_yaml", "updated_yaml"),
    [
        pytest.param(
            {
                "artifacts": {
                    "model": {
                        "path": "model.pth",
                        "type": "model",
                        "desc": "best model",
                    },
                },
            },
            {
                "artifacts": {
                    "model": {"path": "dvclive/artifacts/model.pth", "type": "model"},
                },
            },
            id="props",
        ),
        pytest.param(
            {
                "artifacts": {
                    "duplicate": {"path": "dvclive/artifacts/model.pth"},
                },
            },
            {
                "artifacts": {
                    "model": {"path": "dvclive/artifacts/model.pth", "type": "model"},
                },
            },
            id="duplicate",
        ),
        pytest.param(
            {
                "artifacts": {
                    "data": {"path": "data.csv", "desc": "source data"},
                },
            },
            {
                "artifacts": {
                    "model": {"path": "dvclive/artifacts/model.pth", "type": "model"},
                    "data": {"path": "data.csv", "desc": "source data"},
                },
            },
            id="keep_extra",
        ),
    ],
)
def test_make_dvcyaml_update_artifact(
    tmp_dir, mocked_dvc_repo, orig_yaml, updated_yaml
):
    dump_yaml(orig_yaml, "dvc.yaml")
    (tmp_dir / "model.pth").touch()

    live = Live(dvcyaml="dvc.yaml")
    live.log_artifact("model.pth", type="model", copy=True)
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == updated_yaml


def test_make_dvcyaml_update_all(tmp_dir, mocked_dvc_repo):
    orig_yaml = {
        "stages": {"train": {"cmd": "train.py"}},
        "metrics": [
            "dvclive/metrics.json",
            "dvclive/metrics.yaml",
            "other/metrics.json",
        ],
        "params": ["dvclive/params.yaml"],
        "plots": [
            {"dvclive/plots/metrics": {"x": "step", "y": "foo"}},
            "dvclive/plots/images",
            "other/plots",
            {
                "custom": {
                    "x": "step",
                    "y": {"dvclive/plots/metrics": "foo"},
                    "title": "custom",
                }
            },
            {
                "dvclive/plots/sklearn/confusion_matrix.json": {
                    "template": "confusion",
                    "x": "actual",
                    "y": "predicted",
                    "title": "Confusion Matrix",
                    "x_label": "True Label",
                    "y_label": "Predicted Label",
                },
            },
        ],
        "artifacts": {
            "model": {"path": "dvclive/artifacts/model.pth", "type": "model"},
            "duplicate": {"path": "dvclive/artifacts/model.pth"},
            "data": {"path": "data.csv", "desc": "source data"},
            "other": {"path": "other.pth"},
        },
    }

    updated_yaml = {
        "stages": {"train": {"cmd": "train.py"}},
        "metrics": ["other/metrics.json", "dvclive/metrics.json"],
        "plots": [
            "other/plots",
            {
                "custom": {
                    "x": "step",
                    "y": {"dvclive/plots/metrics": "foo"},
                    "title": "custom",
                }
            },
            {"dvclive/plots/metrics": {"x": "step"}},
            "dvclive/plots/images",
        ],
        "artifacts": {
            "model": {"path": "dvclive/artifacts/model.pth", "type": "model"},
            "data": {"path": "data.csv", "desc": "source data"},
            "other": {"path": "other.pth"},
        },
    }

    dump_yaml(orig_yaml, "dvc.yaml")
    (tmp_dir / "model.pth").touch()
    (tmp_dir / "data.csv").touch()

    live = Live(dvcyaml="dvc.yaml")
    live.log_metric("foo", 2)
    live.log_image("img.png", Image.new("RGB", (10, 10), (250, 250, 250)))
    live.log_artifact("model.pth", type="model", copy=True)
    live.log_artifact("data.csv", desc="source data")
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == updated_yaml


def test_make_dvcyaml_update_multiple(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    live = Live("train", dvcyaml="dvc.yaml")
    live.log_metric("foo", 2)
    live.log_artifact("model.pth", type="model", copy=True)
    make_dvcyaml(live)

    live = Live("eval", dvcyaml="dvc.yaml")
    live.log_metric("bar", 3)
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["train/metrics.json", "eval/metrics.json"],
        "plots": [
            {"train/plots/metrics": {"x": "step"}},
            {"eval/plots/metrics": {"x": "step"}},
        ],
        "artifacts": {
            "model": {"path": "train/artifacts/model.pth", "type": "model"},
        },
    }


@pytest.mark.parametrize("dvcyaml", [True, False])
def test_dvcyaml_on_next_step(tmp_dir, dvcyaml, mocked_dvc_repo):
    live = Live(dvcyaml=dvcyaml)
    live.next_step()
    if dvcyaml:
        assert (tmp_dir / live.dvc_file).exists()
    else:
        assert not (tmp_dir / live.dvc_file).exists()


@pytest.mark.parametrize("dvcyaml", [True, False])
def test_dvcyaml_on_end(tmp_dir, dvcyaml, mocked_dvc_repo):
    live = Live(dvcyaml=dvcyaml)
    live.end()
    if dvcyaml:
        assert (tmp_dir / live.dvc_file).exists()
    else:
        assert not (tmp_dir / live.dvc_file).exists()


def test_make_dvcyaml_idempotent(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        live.log_artifact("model.pth", type="model")

    live.make_dvcyaml()

    assert load_yaml(live.dvc_file) == {
        "artifacts": {
            "model": {"path": "model.pth", "type": "model"},
        }
    }


@pytest.mark.parametrize("dvcyaml", [True, False, "dvclive/dvc.yaml"])
def test_warn_on_dvcyaml_output_overlap(tmp_dir, mocker, mocked_dvc_repo, dvcyaml):
    logger = mocker.patch("dvclive.live.logger")
    dvc_stage = mocker.MagicMock()
    dvc_stage.addressing = "train"
    dvc_out = mocker.MagicMock()
    dvc_out.fs_path = tmp_dir / "dvclive"
    dvc_stage.outs = [dvc_out]
    mocked_dvc_repo.index.stages = [dvc_stage]
    live = Live(dvcyaml=dvcyaml)

    if dvcyaml == "dvclive/dvc.yaml":
        msg = f"'{live.dvc_file}' is in outputs of stage 'train'.\n"
        msg += "Remove it from outputs to make DVCLive work as expected."
        logger.warning.assert_called_with(msg)
    else:
        logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "dvcyaml",
    [True, False, "dvc.yaml", Path("dvc.yaml")],
)
def test_make_dvcyaml(tmp_dir, mocked_dvc_repo, dvcyaml):
    dvclive = Live("logs", dvcyaml=dvcyaml)
    dvclive.log_metric("m1", 1)
    dvclive.next_step()

    if dvcyaml:
        assert "metrics" in load_yaml(dvclive.dvc_file)
    else:
        assert not os.path.exists(dvclive.dvc_file)

    dvclive.make_dvcyaml()
    assert "metrics" in load_yaml(dvclive.dvc_file)


def test_make_dvcyaml_no_repo(tmp_dir, mocker):
    dvclive = Live("logs")
    dvclive.make_dvcyaml()

    assert os.path.exists("dvc.yaml")


def test_make_dvcyaml_invalid(tmp_dir, mocker):
    with pytest.raises(InvalidDvcyamlError):
        Live("logs", dvcyaml="invalid")


def test_make_dvcyaml_on_end(tmp_dir, mocker):
    dvclive = Live("logs")
    dvclive.end()

    assert os.path.exists("dvc.yaml")


def test_make_dvcyaml_false(tmp_dir, mocker):
    dvclive = Live("logs", dvcyaml=False)
    dvclive.end()

    assert not os.path.exists("dvc.yaml")


def test_make_dvcyaml_none(tmp_dir, mocker):
    dvclive = Live("logs", dvcyaml=None)
    dvclive.end()

    assert not os.path.exists("dvc.yaml")
