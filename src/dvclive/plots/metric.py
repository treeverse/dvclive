import csv
import os
import time
from pathlib import Path

from .base import Data
from .utils import NUMPY_SCALARS


class Metric(Data):
    suffixes = (".csv", ".tsv")
    subfolder = "metrics"

    @staticmethod
    def could_log(val: object) -> bool:
        if isinstance(val, (int, float, str)):
            return True
        return (
            val.__class__.__module__ == "numpy"
            and val.__class__.__name__ in NUMPY_SCALARS
        )

    @property
    def output_path(self) -> Path:
        _path = Path(f"{self.output_folder / self.name}.tsv")
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    def dump(self, val, **kwargs) -> None:
        row = {}
        if kwargs.get("timestamp", False):
            row["timestamp"] = int(time.time() * 1000)
        row["step"] = self.step
        row[os.path.basename(self.name)] = val

        existed = self.output_path.exists()
        with open(self.output_path, "a", encoding="utf-8", newline="") as fobj:
            writer = csv.DictWriter(
                fobj, row.keys(), delimiter="\t", lineterminator=os.linesep
            )
            if not existed:
                writer.writeheader()
            writer.writerow(row)

    @property
    def summary_keys(self) -> list[str]:
        return os.path.normpath(self.name).split(os.path.sep)
