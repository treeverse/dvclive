# ruff: noqa: ARG002
from typing import Optional

import tensorflow as tf

from dvclive import Live
from dvclive.utils import standardize_metric_name


class DVCLiveCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        save_weights_only: bool = False,
        live: Optional[Live] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_weights_only = save_weights_only
        self.live = live if live is not None else Live(**kwargs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        for metric, value in logs.items():
            self.live.log_metric(standardize_metric_name(metric, __name__), value)
        self.live.next_step()

    def on_train_end(self, logs: Optional[dict] = None):
        self.live.end()
