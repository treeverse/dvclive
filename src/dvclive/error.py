from typing import Any


class DvcLiveError(Exception):
    pass


class InvalidDataTypeError(DvcLiveError):
    def __init__(self, name, val):
        self.name = name
        self.val = val
        super().__init__(f"Data '{name}' has not supported type {val}")


class InvalidDvcyamlError(DvcLiveError):
    def __init__(self):
        super().__init__("`dvcyaml` path must have filename 'dvc.yaml'")


class InvalidImageNameError(DvcLiveError):
    def __init__(self, name):
        self.name = name
        super().__init__(f"Cannot log image with name '{name}'")


class InvalidMetricNameError(DvcLiveError):
    def __init__(self, name, parent):
        super().__init__(
            f"Cannot log metric '{name}': parent '{parent}' already has a value"
        )


class InvalidPlotTypeError(DvcLiveError):
    def __init__(self, name):
        from .plots import SKLEARN_PLOTS

        self.name = name
        super().__init__(
            f"Plot type '{name}' is not supported."
            f"\nSupported types are: {list(SKLEARN_PLOTS)}"
        )


class InvalidParameterTypeError(DvcLiveError):
    def __init__(self, msg: Any):
        super().__init__(msg)


class InvalidReportModeError(DvcLiveError):
    def __init__(self, val):
        super().__init__(
            f"`report` can only be `None`, `auto`, `html`, `notebook` or `md`. "
            f"Got {val} instead."
        )
