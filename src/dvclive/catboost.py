from urllib.parse import quote

from dvclive import Live


class DVCLiveCallback:
    """Log CatBoost CPU training metrics to a Live context managed by the caller."""

    def __init__(self, live: Live):
        self.live = live
        self._history_sizes: dict[tuple[str, str], int] = {}

    def after_iteration(self, info) -> bool:
        if info.iteration == 1:
            self._history_sizes.clear()
        for dataset, metrics in info.metrics.items():
            for name, values in metrics.items():
                key = (dataset, name)
                # CatBoost repeats the history when a metric was not evaluated.
                if len(values) > self._history_sizes.get(key, 0):
                    metric = quote(name, safe="=;,")
                    self.live.log_metric(f"{dataset}/{metric}", values[-1])
                    self._history_sizes[key] = len(values)
        self.live.next_step()
        return True
