import logging
from pathlib import Path
from typing import Optional, Union

from dvclive.error import InvalidDataTypeError
from dvclive.serialize import dump_json
from dvclive.utils import convert_datapoints_to_list_of_dicts

from .base import Data

logger = logging.getLogger(__name__)


class CustomPlot(Data):
    suffixes = (".json",)
    subfolder = "custom"

    def __init__(
        self,
        name: str,
        output_folder: str,
        x: str,
        y: Union[str, list[str]],
        template: Optional[str],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> None:
        super().__init__(name, output_folder)
        self.name = self.name.replace(".json", "")
        if not template:
            template = None

        config = {
            "template": template,
            "x": x,
            "y": y,
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
        }
        self._plot_config = {k: v for k, v in config.items() if v is not None}

    @property
    def output_path(self) -> Path:
        _path = Path(f"{self.output_folder / self.name}.json")
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    @staticmethod
    def could_log(val: object) -> bool:
        return isinstance(val, list) and all(isinstance(x, dict) for x in val)

    def prepare_datapoints(self, datapoints) -> list[dict]:
        config = self.plot_config
        template = config.get("template")
        horizontal = template in ("bar_horizontal", "bar_horizontal_sorted")
        x, y = config["x"], config["y"]
        if horizontal and isinstance(datapoints, dict):
            if not isinstance(y, str) or x == y:
                msg = "Horizontal bar mappings require distinct x and y field names."
                raise ValueError(msg)
            records = [{y: label, x: value} for label, value in datapoints.items()]
        else:
            records = convert_datapoints_to_list_of_dicts(datapoints)

        if not CustomPlot.could_log(records):
            raise InvalidDataTypeError(self.name, type(records))

        if horizontal and records:
            self._check_bar_lengths(records, x, template)
        return records

    def _check_bar_lengths(self, records, x, template):
        # JSON object keys become strings, including integer array column indices.
        values = (
            value for row in records for field, value in row.items() if str(field) == x
        )
        found = False
        for value in values:
            found = True
            if value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError, OverflowError):
                logger.warning(
                    "Plot %r uses %s, which expects numeric values for x field %r. "
                    "Check that x contains bar lengths and y contains labels.",
                    self.name,
                    template,
                    x,
                )
                return
        if not found:
            logger.warning(
                "Plot %r uses %s, but no row has x field %r.", self.name, template, x
            )

    @property
    def plot_config(self):
        return self._plot_config

    def dump(self, val, **kwargs) -> None:  # noqa: ARG002
        dump_json(val, self.output_path)
