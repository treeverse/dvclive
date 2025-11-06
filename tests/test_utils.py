import numpy as np
import pandas as pd
import pytest

from dvclive.error import InvalidDataTypeError
from dvclive.utils import convert_datapoints_to_list_of_dicts, standardize_metric_name


@pytest.mark.parametrize(
    ("framework", "logged", "standardized"),
    [
        ("dvclive.lightning", "epoch", "epoch"),
        ("dvclive.lightning", "train_loss", "train/loss"),
        ("dvclive.lightning", "train_loss_epoch", "train/epoch/loss"),
        ("dvclive.lightning", "train_model_error", "train/model_error"),
        ("dvclive.lightning", "grad_step", "grad_step"),
    ],
)
def test_standardize_metric_name(framework, logged, standardized):
    assert standardize_metric_name(logged, framework) == standardized


# Tests for convert_datapoints_to_list_of_dicts()
@pytest.mark.parametrize(
    ("input_data", "expected_output"),
    [
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            [{"A": 1, "B": 3}, {"A": 2, "B": 4}],
        ),
        (np.array([[1, 3], [2, 4]]), [{0: 1, 1: 3}, {0: 2, 1: 4}]),
        (
            np.array([(1, 3), (2, 4)], dtype=[("A", "i4"), ("B", "i4")]),
            [{"A": 1, "B": 3}, {"A": 2, "B": 4}],
        ),
        ([{"A": 1, "B": 3}, {"A": 2, "B": 4}], [{"A": 1, "B": 3}, {"A": 2, "B": 4}]),
    ],
)
def test_convert_datapoints_to_list_of_dicts(input_data, expected_output):
    assert convert_datapoints_to_list_of_dicts(input_data) == expected_output


def test_unsupported_format():
    with pytest.raises(InvalidDataTypeError) as exc_info:
        convert_datapoints_to_list_of_dicts("unsupported data format")

    assert "not supported type" in str(exc_info.value)
