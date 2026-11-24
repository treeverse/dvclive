import json

import pandas as pd
import pytest

from dvclive import Live
from dvclive.error import InvalidDataTypeError
from dvclive.plots.custom import CustomPlot


def test_log_custom_plot(tmp_dir):
    live = Live()
    out = tmp_dir / live.plots_dir / CustomPlot.subfolder

    datapoints = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    live.log_plot(
        "custom_linear",
        datapoints,
        x="x",
        y="y",
        template="linear",
        title="custom_title",
        x_label="x_label",
        y_label="y_label",
    )

    assert json.loads((out / "custom_linear.json").read_text()) == datapoints
    assert live._plots["custom_linear"].plot_config == {
        "template": "linear",
        "title": "custom_title",
        "x": "x",
        "y": "y",
        "x_label": "x_label",
        "y_label": "y_label",
    }


def test_log_custom_plot_multi_y(tmp_dir):
    live = Live()
    out = tmp_dir / live.plots_dir / CustomPlot.subfolder

    datapoints = [{"x": 1, "y1": 2, "y2": 3}, {"x": 4, "y1": 5, "y2": 6}]
    live.log_plot(
        "custom_linear",
        datapoints,
        x="x",
        y=["y1", "y2"],
        template="linear",
        title="custom_title",
        x_label="x_label",
        y_label="y_label",
    )

    assert json.loads((out / "custom_linear.json").read_text()) == datapoints
    assert live._plots["custom_linear"].plot_config == {
        "template": "linear",
        "title": "custom_title",
        "x": "x",
        "y": ["y1", "y2"],
        "x_label": "x_label",
        "y_label": "y_label",
    }


def test_log_custom_plot_with_template_as_empty_string(tmp_dir):
    live = Live()
    out = tmp_dir / live.plots_dir / CustomPlot.subfolder

    datapoints = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    live.log_plot(
        "custom_linear",
        datapoints,
        x="x",
        y="y",
        template="",
        title="custom_title",
        x_label="x_label",
        y_label="y_label",
    )

    assert json.loads((out / "custom_linear.json").read_text()) == datapoints
    # 'template' should not be in plot_config. Default template will be assigned later.
    assert live._plots["custom_linear"].plot_config == {
        "title": "custom_title",
        "x": "x",
        "y": "y",
        "x_label": "x_label",
        "y_label": "y_label",
    }


@pytest.mark.parametrize("template", ["bar_horizontal", "bar_horizontal_sorted"])
def test_horizontal_bar_mapping(tmp_dir, template, caplog):
    live = Live(report=None)
    weights = {"petal_width": 0.4, "sepal_length": 0.03}
    live.log_plot("weights", weights, x="importance", y="feature", template=template)
    output = tmp_dir / live.plots_dir / CustomPlot.subfolder / "weights.json"
    assert json.loads(output.read_text()) == [
        {"feature": "petal_width", "importance": 0.4},
        {"feature": "sepal_length", "importance": 0.03},
    ]
    assert "numeric" not in caplog.text


def test_horizontal_bar_warns_for_swapped_axes(tmp_dir, caplog):
    live = Live(report=None)
    frame = pd.DataFrame({"feature": ["width", "length"], "importance": [0.4, 0.03]})
    live.log_plot(
        "weights", frame, x="feature", y="importance", template="bar_horizontal"
    )

    assert "numeric" in caplog.text
    assert "feature" in caplog.text
    assert "bar_horizontal" in caplog.text
    output = tmp_dir / live.plots_dir / CustomPlot.subfolder / "weights.json"
    assert json.loads(output.read_text()) == frame.to_dict(orient="records")


def test_mapping_requires_horizontal_bar(tmp_dir):
    live = Live(report=None)
    with pytest.raises(InvalidDataTypeError):
        live.log_plot("weights", {"width": 0.4}, x="importance", y="feature")


@pytest.mark.parametrize("y", ["importance", ["feature", "group"]])
def test_mapping_requires_distinct_fields(tmp_dir, y):
    live = Live(report=None)
    with pytest.raises(ValueError, match="distinct x and y"):
        live.log_plot(
            "weights", {"width": 0.4}, x="importance", y=y, template="bar_horizontal"
        )
    assert "weights" not in live._plots


def test_horizontal_bar_missing_column(tmp_dir, caplog):
    live = Live(report=None)
    live.log_plot(
        "weights", [{"label": "width"}], x="value", y="label", template="bar_horizontal"
    )
    assert "no row has x field 'value'" in caplog.text


def test_horizontal_bar_numeric_column_reused_plot(tmp_dir, caplog):
    live = Live(report=None)
    live.log_plot(
        "weights",
        [{0: None, 1: "width"}, {0: "0.3", 1: "length"}],
        x="0",
        y="1",
        template="bar_horizontal",
    )
    # Subsequent writes use the plot configuration recorded on the first call.
    live.log_plot("weights", {"depth": 0.1}, x="0", y="1")
    assert "x field" not in caplog.text
    output = tmp_dir / live.plots_dir / CustomPlot.subfolder / "weights.json"
    assert json.loads(output.read_text()) == [{"1": "depth", "0": 0.1}]
