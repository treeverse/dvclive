import time
import unittest
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pytest
from dvc.env import DVC_EXP_GIT_REMOTE
from dvc_studio_client import DEFAULT_STUDIO_URL
from dvc_studio_client.env import DVC_STUDIO_REPO_URL, DVC_STUDIO_TOKEN
from PIL import Image as ImagePIL

from dvclive import Live
from dvclive.env import DVC_EXP_BASELINE_REV, DVC_EXP_NAME, DVC_ROOT
from dvclive.plots import Image, Metric
from dvclive.studio import _adapt_image, get_dvc_studio_config, post_to_studio


def get_studio_call(event_type, exp_name, **kwargs):
    data = {
        "type": event_type,
        "name": exp_name,
        "repo_url": "STUDIO_REPO_URL",
        "baseline_sha": kwargs.pop("baseline_sha", None) or "f" * 40,
        "client": "dvclive",
    } | kwargs

    return {
        "json": data,
        "headers": {
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        "timeout": (30, 5),
    }


def test_post_to_studio(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    live = Live()
    live.log_param("fooparam", 1)

    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    mocked_post, _ = mocked_studio_post

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live", **get_studio_call("start", exp_name=live._exp_name)
    )

    live.log_metric("foo", 1)
    live.step = 0
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            exp_name=live._exp_name,
            step=0,
            plots={f"{foo_path}": {"data": [{"step": 0, "foo": 1.0}]}},
        ),
    )

    live.step += 1
    live.log_metric("foo", 2)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            exp_name=live._exp_name,
            step=1,
            plots={f"{foo_path}": {"data": [{"step": 1, "foo": 2.0}]}},
        ),
    )

    mocked_post.reset_mock()
    live.save_dvc_exp()
    data = live._get_live_data()
    post_to_studio(live, "done", data)

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "done", exp_name=live._exp_name, experiment_rev=live._experiment_rev
        ),
    )


def test_post_to_studio_subrepo(tmp_dir, mocked_dvc_subrepo, mocked_studio_post):
    live = Live()
    live.log_param("fooparam", 1)

    mocked_post, _ = mocked_studio_post

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call("start", exp_name=live._exp_name, subdir="subdir"),
    )


def test_post_to_studio_repo_url(tmp_dir, dvc_repo, mocked_studio_post, monkeypatch):
    monkeypatch.setenv(DVC_EXP_GIT_REMOTE, "dvc_exp_git_remote")

    live = Live()
    live.log_param("fooparam", 1)

    mocked_post, _ = mocked_studio_post

    assert mocked_post.call_args.kwargs["json"]["repo_url"] == "dvc_exp_git_remote"


def test_post_to_studio_failed_data_request(
    tmp_dir, mocker, mocked_dvc_repo, mocked_studio_post
):
    mocked_post, valid_response = mocked_studio_post

    live = Live()

    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    error_response = mocker.MagicMock()
    error_response.status_code = 400
    mocker.patch("requests.post", return_value=error_response)
    live.log_metric("foo", 1)
    live.step = 0
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    live.step += 1
    live.log_metric("foo", 2)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)
    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            exp_name=live._exp_name,
            step=1,
            plots={
                f"{foo_path}": {
                    "data": [{"step": 0, "foo": 1.0}, {"step": 1, "foo": 2.0}]
                }
            },
        ),
    )


def test_post_to_studio_failed_start_request(
    tmp_dir, mocker, mocked_dvc_repo, mocked_studio_post
):
    mocked_response = mocker.MagicMock()
    mocked_response.status_code = 400
    mocked_post = mocker.patch("requests.post", return_value=mocked_response)

    live = Live()

    live.log_metric("foo", 1)
    live.next_step()

    live.log_metric("foo", 2)
    live.next_step()

    assert mocked_post.call_count == 1
    assert live._studio_events_to_skip == {"start", "data", "done"}


def test_post_to_studio_done_only_once(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    mocked_post, _ = mocked_studio_post
    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    expected_done_calls = [
        call
        for call in mocked_post.call_args_list
        if call.kwargs["json"]["type"] == "done"
    ]
    live.end()
    actual_done_calls = [
        call
        for call in mocked_post.call_args_list
        if call.kwargs["json"]["type"] == "done"
    ]
    assert expected_done_calls == actual_done_calls


def test_post_to_studio_snapshots_data_to_send(
    tmp_dir, mocked_dvc_repo, mocked_studio_post
):
    # Tests race condition between main app thread and Studio post thread
    # where the main thread can be faster in producing metrics than the
    # Studio post thread in sending them.
    mocked_post, _ = mocked_studio_post

    calls = defaultdict(dict)

    def _long_post(*_, **kwargs):
        if kwargs["json"]["type"] == "data":
            # Mock by default doesn't copy lists, dict, we share "body" var in
            # some calls, thus we can't rely on `mocked_post.call_args_list`
            json = deepcopy(kwargs)["json"]
            step = json["step"]
            for key in ["metrics", "params", "plots"]:
                if key in json:
                    calls[step][key] = json[key]
            time.sleep(0.1)
        return unittest.mock.DEFAULT

    mocked_post.side_effect = lambda *args, **kwargs: _long_post(*args, **kwargs)

    live = Live()
    for i in range(10):
        live.log_metric("foo", i)
        live.log_param(f"fooparam-{i}", i)
        live.log_image(f"foo.{i}.png", ImagePIL.new("RGB", (i + 1, i + 1), (0, 0, 0)))
        live.next_step()

    live._wait_for_studio_updates_posted()

    assert len(calls) == 10
    for i in range(10):
        call = calls[i]
        assert call["metrics"] == {
            "dvclive/metrics.json": {"data": {"foo": i, "step": i}}
        }
        assert call["params"] == {
            "dvclive/params.yaml": {f"fooparam-{k}": k for k in range(i + 1)}
        }
        # Check below that `plots`` has the following shape
        # {
        #    'dvclive/plots/metrics/foo.tsv': {'data': [{'step': i, 'foo': float(i)}]},
        #    f"dvclive/plots/images/foo.{i}.png": {'image': '...'}
        # }
        assert len(call["plots"]) == 2
        foo_data = call["plots"]["dvclive/plots/metrics/foo.tsv"]["data"]
        assert len(foo_data) == 1
        assert foo_data[0]["step"] == i
        assert foo_data[0]["foo"] == pytest.approx(float(i))
        assert call["plots"][f"dvclive/plots/images/foo.{i}.png"]["image"]


def test_studio_updates_posted_on_end(tmp_path, mocked_dvc_repo, mocked_studio_post):
    mocked_post, valid_response = mocked_studio_post
    metrics_file = tmp_path / "metrics.json"
    metrics_content = "metrics"

    def long_post(*args, **kwargs):
        # in case of `data` `long_post` should be called from a separate thread,
        # meanwhile main thread go forward without slowing down, so if there is no
        # some kind of wait in the Live main thread, then it will complete before
        # we even can have a chance to write the file below
        if kwargs["json"]["type"] == "data":
            time.sleep(1)
            metrics_file.write_text(metrics_content)

        return valid_response

    mocked_post.side_effect = long_post

    with Live() as live:
        live.log_metric("foo", 1)

    assert metrics_file.read_text() == metrics_content


def test_studio_update_raises_exception(tmp_path, mocked_dvc_repo, mocked_studio_post):
    # Test that if a studio update raises an exception, main process doesn't hang on
    # queue join in the Live main thread.
    # https://github.com/iterative/dvclive/pull/864
    mocked_post, valid_response = mocked_studio_post

    def post_raises_exception(*args, **kwargs):
        if kwargs["json"]["type"] == "data":
            # We'll hit this sleep only once, other calls are ignored
            # after the exception is raised
            time.sleep(1)
            raise Exception("test exception")  # noqa: TRY002, TRY003
        return valid_response

    mocked_post.side_effect = post_raises_exception

    with Live() as live:
        live.log_metric("foo", 1)
        live.log_metric("foo", 2)
        live.log_metric("foo", 3)

    # Only 1 data call is made, other calls are ignored after the exception is raised
    assert mocked_post.call_count == 3
    assert [e.kwargs["json"]["type"] for e in mocked_post.call_args_list] == [
        "start",
        "data",
        "done",
    ]


@pytest.mark.studio
def test_post_to_studio_skip_start_and_done_on_env_var(
    tmp_dir, mocked_dvc_repo, mocked_studio_post, monkeypatch
):
    mocked_post, _ = mocked_studio_post

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")
    monkeypatch.setenv(DVC_ROOT, tmp_dir)

    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    call_types = [call.kwargs["json"]["type"] for call in mocked_post.call_args_list]
    assert "start" not in call_types
    assert "done" not in call_types


@pytest.mark.studio
def test_post_to_studio_dvc_studio_config(
    tmp_dir, mocker, mocked_dvc_repo, mocked_studio_post, monkeypatch
):
    mocked_post, _ = mocked_studio_post

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")
    monkeypatch.setenv(DVC_ROOT, tmp_dir)
    monkeypatch.delenv(DVC_STUDIO_TOKEN)

    mocked_dvc_repo.config = {"studio": {"token": "token"}}

    with Live() as live:
        live.log_metric("foo", 1)
        live.step = 0
        live.make_summary()
        data = live._get_live_data()
        post_to_studio(live, "data", data)

    assert mocked_post.call_args.kwargs["headers"]["Authorization"] == "token token"


@pytest.mark.studio
def test_post_to_studio_skip_if_no_token(
    tmp_dir,
    mocker,
    monkeypatch,
    mocked_dvc_repo,
):
    mocked_post = mocker.patch("dvclive.studio.post_live_metrics", return_value=None)

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    mocked_dvc_repo.config = {}

    with Live() as live:
        live.log_metric("foo", 1)
        live.step = 0
        live.make_summary()
        data = live._get_live_data()
        post_to_studio(live, "data", data)

    assert mocked_post.call_count == 0


def test_post_to_studio_shorten_names(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    mocked_post, _ = mocked_studio_post

    live = Live()
    live.log_metric("eval/loss", 1)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    plots_path = Path(live.plots_dir)
    loss_path = (plots_path / Metric.subfolder / "eval/loss.tsv").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            exp_name=live._exp_name,
            step=0,
            plots={f"{loss_path}": {"data": [{"step": 0, "loss": 1.0}]}},
        ),
    )


@pytest.mark.studio
def test_post_to_studio_inside_dvc_exp(
    tmp_dir, mocker, monkeypatch, mocked_studio_post, mocked_dvc_repo
):
    mocked_post, _ = mocked_studio_post

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")
    monkeypatch.setenv(DVC_ROOT, tmp_dir)

    with Live() as live:
        live.log_metric("foo", 1)
        live.step = 0
        live.make_summary()
        data = live._get_live_data()
        post_to_studio(live, "data", data)

    call_types = [call.kwargs["json"]["type"] for call in mocked_post.call_args_list]
    assert "start" not in call_types
    assert "done" not in call_types


@pytest.mark.studio
def test_post_to_studio_inside_subdir(
    tmp_dir, dvc_repo, mocker, monkeypatch, mocked_studio_post, mocked_dvc_repo
):
    mocked_post, _ = mocked_studio_post
    subdir = tmp_dir / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    live = Live()
    live.log_metric("foo", 1)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            baseline_sha=live._baseline_rev,
            exp_name=live._exp_name,
            step=0,
            plots={f"subdir/{foo_path}": {"data": [{"step": 0, "foo": 1.0}]}},
        ),
    )


@pytest.mark.studio
def test_post_to_studio_inside_subdir_dvc_exp(
    tmp_dir, dvc_repo, monkeypatch, mocked_studio_post, mocked_dvc_repo
):
    mocked_post, _ = mocked_studio_post
    subdir = tmp_dir / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    live = Live()
    live.log_metric("foo", 1)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            baseline_sha=live._baseline_rev,
            exp_name=live._exp_name,
            step=0,
            plots={f"subdir/{foo_path}": {"data": [{"step": 0, "foo": 1.0}]}},
        ),
    )


def test_post_to_studio_without_exp(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    assert not Live(save_dvc_exp=False)._studio_events_to_skip


def test_get_dvc_studio_config_none(mocker):
    mocker.patch("dvclive.live.get_dvc_repo", return_value=None)
    live = Live()
    assert get_dvc_studio_config(live) == {}


def test_get_dvc_studio_config_env_var(monkeypatch, mocker):
    monkeypatch.setenv(DVC_STUDIO_TOKEN, "token")
    monkeypatch.setenv(DVC_STUDIO_REPO_URL, "repo_url")
    mocker.patch("dvclive.live.get_dvc_repo", return_value=None)
    live = Live()
    assert get_dvc_studio_config(live) == {
        "token": "token",
        "repo_url": "repo_url",
        "url": DEFAULT_STUDIO_URL,
    }


def test_get_dvc_studio_config_dvc_repo(mocked_dvc_repo):
    mocked_dvc_repo.config = {"studio": {"token": "token", "repo_url": "repo_url"}}
    live = Live()
    assert get_dvc_studio_config(live) == {
        "token": "token",
        "repo_url": "repo_url",
        "url": DEFAULT_STUDIO_URL,
    }


def test_post_to_studio_images(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    mocked_post, _ = mocked_studio_post

    live = Live()
    live.log_image("foo.png", ImagePIL.new("RGB", (10, 10), (0, 0, 0)))
    live.step = 0
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    foo_path = (Path(live.plots_dir) / Image.subfolder / "foo.png").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            baseline_sha=live._baseline_rev,
            exp_name=live._exp_name,
            step=0,
            plots={f"{foo_path}": {"image": _adapt_image(foo_path)}},
        ),
    )


def test_post_to_studio_message(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    live = Live(exp_message="Custom message")

    mocked_post, _ = mocked_studio_post

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call("start", exp_name=live._exp_name, message="Custom message"),
    )


def test_post_to_studio_name(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    Live(exp_name="custom-name")

    mocked_post, _ = mocked_studio_post

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call("start", exp_name="custom-name"),
    )


def test_post_to_studio_if_done_skipped(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    with Live() as live:
        live._studio_events_to_skip.add("start")
        live._studio_events_to_skip.add("done")
        live.log_metric("foo", 1)
        live.step = 0
        live.make_summary()
        data = live._get_live_data()
        post_to_studio(live, "data", data)

    mocked_post, _ = mocked_studio_post
    call_types = [call.kwargs["json"]["type"] for call in mocked_post.call_args_list]
    assert "data" in call_types


@pytest.mark.studio
def test_post_to_studio_no_repo(tmp_dir, monkeypatch, mocked_studio_post):
    monkeypatch.setenv(DVC_STUDIO_TOKEN, "STUDIO_TOKEN")
    monkeypatch.setenv(DVC_STUDIO_REPO_URL, "STUDIO_REPO_URL")

    live = Live(save_dvc_exp=True)
    live.log_param("fooparam", 1)

    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    mocked_post, _ = mocked_studio_post

    mocked_post.assert_called()
    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call("start", baseline_sha="0" * 40, exp_name=live._exp_name),
    )

    live.log_metric("foo", 1)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            baseline_sha="0" * 40,
            exp_name=live._exp_name,
            step=0,
            plots={f"{foo_path}": {"data": [{"step": 0, "foo": 1.0}]}},
        ),
    )

    live.step += 1
    live.log_metric("foo", 2)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            baseline_sha="0" * 40,
            exp_name=live._exp_name,
            step=1,
            plots={f"{foo_path}": {"data": [{"step": 1, "foo": 2.0}]}},
        ),
    )

    post_to_studio(live, "done")
    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call("done", baseline_sha="0" * 40, exp_name=live._exp_name),
    )


@pytest.mark.studio
def test_post_to_studio_skip_if_no_repo_url(
    tmp_dir,
    mocker,
    monkeypatch,
):
    mocked_post = mocker.patch("dvclive.studio.post_live_metrics", return_value=None)

    monkeypatch.setenv(DVC_STUDIO_TOKEN, "token")

    with Live() as live:
        live.log_metric("foo", 1)
        live.step = 0
        live.make_summary()
        data = live._get_live_data()
        post_to_studio(live, "data", data)

    assert mocked_post.call_count == 0


def test_post_to_studio_repeat_step(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    # for more context see the PR https://github.com/iterative/dvclive/pull/788
    live = Live()

    prefix = Path(live.plots_dir) / Metric.subfolder
    foo_path = (prefix / "foo.tsv").as_posix()
    bar_path = (prefix / "bar.tsv").as_posix()

    mocked_post, _ = mocked_studio_post

    live.step = 0
    live.log_metric("foo", 1)
    live.log_metric("bar", 0.1)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            exp_name=live._exp_name,
            step=0,
            plots={
                f"{foo_path}": {"data": [{"step": 0, "foo": 1.0}]},
                f"{bar_path}": {"data": [{"step": 0, "bar": 0.1}]},
            },
        ),
    )

    live.log_metric("foo", 2)
    live.log_metric("foo", 3)
    live.log_metric("bar", 0.2)
    live.make_summary()
    data = live._get_live_data()
    post_to_studio(live, "data", data)

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        **get_studio_call(
            "data",
            exp_name=live._exp_name,
            step=0,
            plots={
                f"{foo_path}": {
                    "data": [{"step": 0, "foo": 2.0}, {"step": 0, "foo": 3.0}]
                },
                f"{bar_path}": {"data": [{"step": 0, "bar": 0.2}]},
            },
        ),
    )
