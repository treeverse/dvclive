from pathlib import Path

import pytest

from dvclive import Live
from dvclive.utils import parse_tsv

catboost = pytest.importorskip("catboost")


@pytest.mark.parametrize(
    ("n_eval_sets", "metric_period", "steps"),
    [(0, 1, [0, 1, 2, 3]), (2, 2, [0, 2, 3])],
)
def test_catboost_metrics(tmp_dir, n_eval_sets, metric_period, steps):
    from dvclive.catboost import DVCLiveCallback

    data = [[0], [1], [2], [3], [4], [5]]
    labels = [0, 0, 1, 0, 1, 1]
    model = catboost.CatBoostClassifier(
        iterations=4,
        metric_period=metric_period,
        custom_metric="F:beta=2",
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    with Live(report=None, save_dvc_exp=False) as live:
        model.fit(
            data,
            labels,
            eval_set=[(data, labels)] * n_eval_sets or None,
            early_stopping_rounds=5 if n_eval_sets else None,
            callbacks=[DVCLiveCallback(live)],
        )

    results = model.get_evals_result()
    assert set(results) == {"learn", *(f"validation_{i}" for i in range(n_eval_sets))}
    for dataset, metrics in results.items():
        for name, expected in metrics.items():
            filename = name.replace(":", "%3A")
            path = Path(live.plots_dir) / "metrics" / dataset / f"{filename}.tsv"
            rows = parse_tsv(path)
            # CatBoost evaluates the early-stopping metric on every iteration.
            expected_steps = (
                [0, 1, 2, 3]
                if (dataset, name) == ("validation_1", "Logloss")
                else steps
            )
            assert [int(row["step"]) for row in rows] == expected_steps
            assert [float(row[filename]) for row in rows] == pytest.approx(expected)
