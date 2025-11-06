import numpy as np
import pytest
from PIL import Image

from dvclive import Live
from dvclive.env import DVCLIVE_OPEN
from dvclive.error import InvalidReportModeError
from dvclive.plots import CustomPlot, Metric
from dvclive.plots import Image as LiveImage
from dvclive.plots.sklearn import SKLearnPlot
from dvclive.report import (
    get_custom_plot_renderers,
    get_image_renderers,
    get_metrics_renderers,
    get_params_renderers,
    get_scalar_renderers,
    get_sklearn_plot_renderers,
)


@pytest.mark.parametrize("mode", ["html", "md", "notebook"])
def test_get_image_renderers(tmp_dir, mode):
    with Live() as live:
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        live.log_image("image.png", img)

    image_renderers = get_image_renderers(
        tmp_dir / live.plots_dir / LiveImage.subfolder
    )
    assert len(image_renderers) == 1
    img = image_renderers[0].datapoints[0]
    assert img["src"].startswith("data:image;base64,")
    assert img["rev"] == "image.png"


def test_get_renderers(tmp_dir, mocker):
    live = Live()

    live.log_param("string", "goo")
    live.log_param("number", 2)

    for i in range(2):
        live.log_metric("foo/bar", i)
        live.next_step()

    scalar_renderers = get_scalar_renderers(tmp_dir / live.plots_dir / Metric.subfolder)
    assert len(scalar_renderers) == 1
    assert scalar_renderers[0].datapoints == [
        {
            "bar": "0",
            "rev": "workspace",
            "step": "0",
        },
        {
            "bar": "1",
            "rev": "workspace",
            "step": "1",
        },
    ]
    assert scalar_renderers[0].properties["y"] == "bar"
    assert scalar_renderers[0].properties["title"] == "foo/bar"
    assert scalar_renderers[0].name == "static/foo/bar"

    metrics_renderer = get_metrics_renderers(live.metrics_file)[0]
    assert metrics_renderer.datapoints == [{"step": 1, "foo": {"bar": 1}}]

    params_renderer = get_params_renderers(live.params_file)[0]
    assert params_renderer.datapoints == [{"string": "goo", "number": 2}]


def test_report_init(monkeypatch, mocker):
    mocker.patch("dvclive.live.inside_notebook", return_value=False)
    live = Live(report="notebook")
    assert live._report_mode is None

    mocker.patch("dvclive.live.matplotlib_installed", return_value=False)
    live = Live(report="md")
    assert live._report_mode is None

    mocker.patch("dvclive.live.matplotlib_installed", return_value=True)
    live = Live(report="md")
    assert live._report_mode == "md"

    live = Live(report="html")
    assert live._report_mode == "html"

    with pytest.raises(InvalidReportModeError, match="Got foo instead\\."):
        Live(report="foo")


@pytest.mark.parametrize("mode", ["html", "md"])
def test_make_report(tmp_dir, mode):
    last_report = ""
    live = Live(report=mode)
    for i in range(3):
        live.log_metric("foobar", i)
        live.log_metric("foo/bar", i)
        live.make_report()
        live.next_step()
        assert (tmp_dir / live.report_file).exists()
        current_report = (tmp_dir / live.report_file).read_text()
        assert last_report != current_report
        last_report = current_report


@pytest.mark.vscode
def test_make_report_open(tmp_dir, mocker, monkeypatch):
    mocked_open = mocker.patch("webbrowser.open")
    live = Live()
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
    live.make_report()
    live.make_report()

    assert not mocked_open.called

    live = Live(report="html")
    live.log_metric("foo", 1)
    live.next_step()

    assert not mocked_open.called

    monkeypatch.setenv(DVCLIVE_OPEN, "true")

    live = Live(report="html")
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
    live.make_report()

    mocked_open.assert_called_once()


def test_get_plot_renderers_sklearn(tmp_dir):
    live = Live()

    for _ in range(2):
        live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
        live.log_sklearn_plot(
            "confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1], name="train/cm"
        )
        live.log_sklearn_plot("roc", [0, 0, 1, 1], [1, 0.1, 0, 1], name="roc_curve")
        live.log_sklearn_plot(
            "roc", [0, 0, 1, 1], [1, 0.1, 0, 1], name="other_roc.json"
        )
        live.next_step()

    plot_renderers = get_sklearn_plot_renderers(
        tmp_dir / live.plots_dir / SKLearnPlot.subfolder, live
    )
    assert len(plot_renderers) == 4
    plot_renderers_dict = {
        plot_renderer.name: plot_renderer for plot_renderer in plot_renderers
    }
    for name in ("roc_curve", "other_roc"):
        plot_renderer = plot_renderers_dict[name]
        assert plot_renderer.datapoints == [
            {"fpr": 0.0, "rev": "workspace", "threshold": np.inf, "tpr": 0.0},
            {"fpr": 0.5, "rev": "workspace", "threshold": 1.0, "tpr": 0.5},
            {"fpr": 1.0, "rev": "workspace", "threshold": 0.1, "tpr": 0.5},
            {"fpr": 1.0, "rev": "workspace", "threshold": 0.0, "tpr": 1.0},
        ]
        assert plot_renderer.properties == live._plots[name].plot_config

    for name in ("confusion_matrix", "train/cm"):
        plot_renderer = plot_renderers_dict[name]
        assert plot_renderer.datapoints == [
            {"actual": "0", "rev": "workspace", "predicted": "1"},
            {"actual": "0", "rev": "workspace", "predicted": "0"},
            {"actual": "1", "rev": "workspace", "predicted": "0"},
            {"actual": "1", "rev": "workspace", "predicted": "1"},
        ]
        assert plot_renderer.properties == live._plots[name].plot_config


def test_get_plot_renderers_custom(tmp_dir):
    live = Live()

    datapoints = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    for _ in range(2):
        live.log_plot("foo_default", datapoints, x="x", y="y")
        live.log_plot(
            "foo_scatter",
            datapoints,
            x="x",
            y="y",
            template="scatter",
        )
        live.next_step()
    plot_renderers = get_custom_plot_renderers(
        tmp_dir / live.plots_dir / CustomPlot.subfolder, live
    )

    assert len(plot_renderers) == 2
    plot_renderers_dict = {
        plot_renderer.name: plot_renderer for plot_renderer in plot_renderers
    }
    for name in ("foo_default", "foo_scatter"):
        plot_renderer = plot_renderers_dict[name]
        assert plot_renderer.datapoints == [
            {"rev": "workspace", "x": 1, "y": 2},
            {"rev": "workspace", "x": 3, "y": 4},
        ]
        assert plot_renderer.properties == live._plots[name].plot_config


def test_report_notebook(tmp_dir, mocker):
    mocker.patch("dvclive.live.inside_notebook", return_value=True)
    mocked_display = mocker.MagicMock()
    mocker.patch("IPython.display.display", return_value=mocked_display)
    live = Live(report="notebook")
    assert live._report_mode == "notebook"
    live.make_report()
    assert mocked_display.update.called
