import shutil
from pathlib import Path

import pytest
from dvc.exceptions import DvcException

from dvclive import Live
from dvclive.error import InvalidDataTypeError
from dvclive.serialize import load_yaml

dvcyaml = """
stages:
  train:
    cmd: python train.py
    outs:
    - data
"""


@pytest.mark.parametrize("cache", [True, False])
def test_log_artifact(tmp_dir, dvc_repo, cache):
    data = tmp_dir / "data"
    data.touch()
    with Live(save_dvc_exp=False) as live:
        live.log_artifact("data", cache=cache)
    assert data.with_suffix(".dvc").exists() is cache
    assert load_yaml(live.dvc_file) == {}


def test_log_artifact_on_existing_dvc_file(tmp_dir, dvc_repo):
    data = tmp_dir / "data"
    data.write_text("foo")
    with Live(save_dvc_exp=False) as live:
        live.log_artifact("data")

    prev_content = data.with_suffix(".dvc").read_text()

    with Live(save_dvc_exp=False) as live:
        data.write_text("bar")
        live.log_artifact("data")

    assert data.with_suffix(".dvc").read_text() != prev_content


def test_log_artifact_twice(tmp_dir, dvc_repo):
    data = tmp_dir / "data"
    with Live(save_dvc_exp=False) as live:
        for i in range(2):
            data.write_text(str(i))
            live.log_artifact("data")
    assert data.with_suffix(".dvc").exists()


def test_log_artifact_with_save_dvc_exp(tmp_dir, mocker, mocked_dvc_repo):
    stage = mocker.MagicMock()
    stage.addressing = "data"
    mocked_dvc_repo.add.return_value = [stage]
    with Live() as live:
        live.log_artifact("data")
    mocked_dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name,
        include_untracked=[live.dir, "data", ".gitignore", "dvc.yaml"],
        force=True,
        message=None,
    )


def test_log_artifact_type_model(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        live.log_artifact("model.pth", type="model")

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "model.pth", "type": "model"}}
    }


def test_log_artifact_dvc_symlink(tmp_dir, dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live(save_dvc_exp=False, dvcyaml="dvc.yaml") as live:
        live._dvc_repo.cache.local.cache_types = ["symlink"]
        live.log_artifact("model.pth", type="model")

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "model.pth", "type": "model"}}
    }


def test_log_artifact_copy(tmp_dir, dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live(save_dvc_exp=False, dvcyaml="dvc.yaml") as live:
        live.log_artifact("model.pth", type="model", copy=True)

    artifacts_dir = Path(live.artifacts_dir)
    assert (artifacts_dir / "model.pth").exists()
    assert (artifacts_dir / "model.pth.dvc").exists()

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "dvclive/artifacts/model.pth", "type": "model"}}
    }


def test_log_artifact_copy_is_not_committed_to_exp(tmp_dir, dvc_repo):
    (tmp_dir / "model.pth").write_text("weights")

    with Live() as live:
        live.log_metric("foo", 1)
        live.log_artifact("model.pth", type="model", copy=True)

    fs = dvc_repo.scm.get_fs(live._experiment_rev)
    assert fs.exists("dvclive/metrics.json")
    assert fs.exists("dvclive/artifacts/model.pth.dvc")
    assert not fs.exists("dvclive/artifacts/model.pth")


def test_log_artifact_copy_overwrite(tmp_dir, dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live(save_dvc_exp=False, dvcyaml="dvc.yaml") as live:
        artifacts_dir = Path(live.artifacts_dir)
        # testing with symlink cache to make sure that DVC protected mode
        # does not prevent the overwrite
        live._dvc_repo.cache.local.cache_types = ["symlink"]
        live.log_artifact("model.pth", type="model", copy=True)
        assert (artifacts_dir / "model.pth").is_symlink()
        live.log_artifact("model.pth", type="model", copy=True)

    assert (artifacts_dir / "model.pth").exists()
    assert (artifacts_dir / "model.pth.dvc").exists()

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "dvclive/artifacts/model.pth", "type": "model"}}
    }


def test_log_artifact_copy_directory_overwrite(tmp_dir, dvc_repo):
    model_path = Path(tmp_dir / "weights")
    model_path.mkdir()
    (tmp_dir / "weights" / "model-epoch-1.pth").touch()

    with Live(save_dvc_exp=False, dvcyaml="dvc.yaml") as live:
        artifacts_dir = Path(live.artifacts_dir)
        # testing with symlink cache to make sure that DVC protected mode
        # does not prevent the overwrite
        live._dvc_repo.cache.local.cache_types = ["symlink"]
        live.log_artifact(model_path, type="model", copy=True)
        assert (artifacts_dir / "weights" / "model-epoch-1.pth").is_symlink()

        shutil.rmtree(model_path)
        model_path.mkdir()
        (tmp_dir / "weights" / "model-epoch-10.pth").write_text("Model weights")
        (tmp_dir / "weights" / "best.pth").write_text("Best model weights")
        live.log_artifact(model_path, type="model", copy=True)

    assert (artifacts_dir / "weights").exists()
    assert (artifacts_dir / "weights" / "best.pth").is_symlink()
    assert (artifacts_dir / "weights" / "best.pth").read_text() == "Best model weights"
    assert (artifacts_dir / "weights" / "model-epoch-10.pth").is_symlink()
    assert len(list((artifacts_dir / "weights").iterdir())) == 2

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"weights": {"path": "dvclive/artifacts/weights", "type": "model"}}
    }


def test_log_artifact_type_model_provided_name(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live(dvcyaml="dvc.yaml") as live:
        live.log_artifact("model.pth", type="model", name="custom")

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"custom": {"path": "model.pth", "type": "model"}}
    }


def test_log_artifact_type_model_on_step_and_final(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live(dvcyaml="dvc.yaml") as live:
        for _ in range(3):
            live.log_artifact("model.pth", type="model")
            live.next_step()
        live.log_artifact("model.pth", type="model", labels=["final"])
    assert load_yaml(live.dvc_file) == {
        "artifacts": {
            "model": {"path": "model.pth", "type": "model", "labels": ["final"]},
        },
        "metrics": ["dvclive/metrics.json"],
    }


def test_log_artifact_type_model_on_step(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live(dvcyaml="dvc.yaml") as live:
        for _ in range(3):
            live.log_artifact("model.pth", type="model")
            live.next_step()

    assert load_yaml(live.dvc_file) == {
        "artifacts": {
            "model": {"path": "model.pth", "type": "model"},
        },
        "metrics": ["dvclive/metrics.json"],
    }


def test_log_artifact_attrs(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    attrs = {
        "type": "model",
        "name": "foo",
        "desc": "bar",
        "labels": ["foo"],
        "meta": {"foo": "bar"},
    }
    with Live(dvcyaml="dvc.yaml") as live:
        live.log_artifact("model.pth", **attrs)
    attrs.pop("name")
    assert load_yaml(live.dvc_file) == {
        "artifacts": {
            "foo": {"path": "model.pth", **attrs},
        }
    }


def test_log_artifact_type_model_when_dvc_add_fails(tmp_dir, mocker, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()
    mocked_dvc_repo.add.side_effect = DvcException("foo")
    with Live(save_dvc_exp=True, dvcyaml="dvc.yaml") as live:
        live.log_artifact("model.pth", type="model")

    assert load_yaml(live.dvc_file) == {
        "artifacts": {"model": {"path": "model.pth", "type": "model"}}
    }


@pytest.mark.parametrize("tracked", ["data_source", "stage", None])
def test_log_artifact_inside_pipeline(tmp_dir, mocker, dvc_repo, tracked):
    logger = mocker.patch("dvclive.live.logger")
    data = tmp_dir / "data"
    data.touch()
    if tracked == "data_source":
        dvc_repo.add(data)
    elif tracked == "stage":
        dvcyaml_path = tmp_dir / "dvc.yaml"
        with open(dvcyaml_path, "w") as f:
            f.write(dvcyaml)
    live = Live(save_dvc_exp=False)
    spy = mocker.spy(live._dvc_repo, "add")
    live._inside_dvc_pipeline = True
    live.log_artifact("data")
    if tracked == "stage":
        msg = (
            "Skipping `dvc add data` because it is already being tracked"
            " automatically as an output of the DVC pipeline."
        )
        logger.info.assert_called_with(msg)
        spy.assert_not_called()
    elif tracked == "data_source":
        msg = (
            "To track 'data' automatically in the DVC pipeline:"
            "\n1. Run `dvc remove data.dvc` "
            "to stop tracking it outside the pipeline."
            "\n2. Add it as an output of the pipeline stage."
        )
        logger.warning.assert_called_with(msg)
        spy.assert_called_once()
    else:
        msg = (
            "To track 'data' automatically in the DVC pipeline, "
            "add it as an output of the pipeline stage."
        )
        logger.warning.assert_called_with(msg)
        spy.assert_called_once()


def test_log_artifact_inside_pipeline_subdir(tmp_dir, mocker, dvc_repo):
    logger = mocker.patch("dvclive.live.logger")
    subdir = tmp_dir / "subdir"
    subdir.mkdir()
    data = subdir / "data"
    data.touch()
    dvc_repo.add(subdir)
    live = Live()
    spy = mocker.spy(live._dvc_repo, "add")
    live._inside_dvc_pipeline = True
    live.log_artifact("subdir/data")
    msg = (
        "To track 'subdir/data' automatically in the DVC pipeline:"
        "\n1. Run `dvc remove subdir.dvc` "
        "to stop tracking it outside the pipeline."
        "\n2. Add it as an output of the pipeline stage."
    )
    logger.warning.assert_called_with(msg)
    spy.assert_called_once()


def test_log_artifact_no_repo(tmp_dir, mocker):
    logger = mocker.patch("dvclive.live.logger")
    (tmp_dir / "data").touch()
    live = Live()
    live.log_artifact("data")
    logger.warning.assert_called_with(
        "A DVC repo is required to log artifacts. Skipping `log_artifact(data)`."
    )


@pytest.mark.parametrize("invalid_path", [None, 1.0, True, [], {}], ids=type)
def test_log_artifact_invalid_path_type(invalid_path, tmp_dir):
    live = Live(save_dvc_exp=False)
    expected_error_msg = f"not supported type {type(invalid_path)}"
    with pytest.raises(InvalidDataTypeError, match=expected_error_msg):
        live.log_artifact(path=invalid_path)
