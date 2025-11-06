# ruff: noqa: SLF001
import base64
import logging
import math
import os
from collections.abc import Mapping
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, Optional

from dvc.exceptions import DvcException
from dvc_studio_client.config import get_studio_config
from dvc_studio_client.post_live_metrics import post_live_metrics

from dvclive.utils import StrPath, rel_path

from .utils import catch_and_warn

if TYPE_CHECKING:
    from dvclive.live import Live
    from dvclive.plots.image import Image

logger = logging.getLogger("dvclive")


def _cast_to_numbers(datapoints: Mapping):
    for datapoint in datapoints:
        for k, v in datapoint.items():
            if k == "step":
                datapoint[k] = int(v)
            elif k == "timestamp":
                continue
            else:
                float_v = float(v)
                if math.isnan(float_v) or math.isinf(float_v):
                    datapoint[k] = str(v)
                else:
                    datapoint[k] = float_v
    return datapoints


def _adapt_path(live: "Live", name: StrPath):
    if live._dvc_repo is not None:
        name = rel_path(name, live._dvc_repo.root_dir)
    if os.name == "nt":
        name = str(PureWindowsPath(name).as_posix())
    return name


def _adapt_image(image_path: StrPath):
    with open(image_path, "rb") as fobj:
        return base64.b64encode(fobj.read()).decode("utf-8")


def _adapt_images(live: "Live", images: "list[Image]"):
    return {
        _adapt_path(live, image.output_path): {"image": _adapt_image(image.output_path)}
        for image in images
        if image.step > live._latest_studio_step
    }


def _get_studio_updates(live: "Live", data: dict[str, Any]):
    params = data["params"]
    plots = data["plots"]
    plots_start_idx = data["plots_start_idx"]
    metrics = data["metrics"]
    images = data["images"]

    params_file = live.params_file
    params_file = _adapt_path(live, params_file)
    params = {params_file: params}

    metrics_file = live.metrics_file
    metrics_file = _adapt_path(live, metrics_file)
    metrics = {metrics_file: {"data": metrics}}

    plots_to_send = {}
    for name, plot in plots.items():
        path = _adapt_path(live, name)
        start_idx = plots_start_idx.get(name, 0)
        num_points_sent = live._num_points_sent_to_studio.get(name, 0)
        plots_to_send[path] = _cast_to_numbers(plot[num_points_sent - start_idx :])

    plots_to_send = {k: {"data": v} for k, v in plots_to_send.items()}
    plots_to_send.update(_adapt_images(live, images))

    return metrics, params, plots_to_send


def get_dvc_studio_config(live: "Live"):
    config = {}
    if live._dvc_repo:
        config = live._dvc_repo.config.get("studio")
    return get_studio_config(dvc_studio_config=config)


def increment_num_points_sent_to_studio(live, plots_sent, data):
    for name in data["plots"]:
        path = _adapt_path(live, name)
        plot = plots_sent.get(path, {})
        if "data" in plot:
            num_points_sent = live._num_points_sent_to_studio.get(name, 0)
            live._num_points_sent_to_studio[name] = num_points_sent + len(plot["data"])
    return live


@catch_and_warn(DvcException, logger)
def post_to_studio(  # noqa: C901
    live: "Live",
    event: Literal["start", "data", "done"],
    data: Optional[dict[str, Any]] = None,
):
    if event in live._studio_events_to_skip:
        return

    kwargs = {}
    if event == "start":
        if message := live._exp_message:
            kwargs["message"] = message
        if subdir := live._subdir:
            kwargs["subdir"] = subdir
    elif event == "data":
        assert data is not None  # noqa: S101
        metrics, params, plots = _get_studio_updates(live, data)
        kwargs["step"] = data["step"]
        kwargs["metrics"] = metrics
        kwargs["params"] = params
        kwargs["plots"] = plots
    elif event == "done" and live._experiment_rev:
        kwargs["experiment_rev"] = live._experiment_rev

    response = post_live_metrics(
        event,
        live._baseline_rev,
        live._exp_name,  # type: ignore[arg-type]
        "dvclive",
        dvc_studio_config=live._dvc_studio_config,
        studio_repo_url=live._repo_url,
        **kwargs,  # type: ignore[arg-type]
    )

    if not response:
        logger.warning(f"`post_to_studio` `{event}` failed.")
        if event == "start":
            live._studio_events_to_skip.add("start")
            live._studio_events_to_skip.add("data")
            live._studio_events_to_skip.add("done")
    elif event == "data":
        assert data is not None  # noqa: S101
        live = increment_num_points_sent_to_studio(live, plots, data)
        live._latest_studio_step = data["step"]

    if event == "done":
        live._studio_events_to_skip.add("done")
        live._studio_events_to_skip.add("data")
