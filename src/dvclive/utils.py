import csv
import json
import os
import re
import shutil
import webbrowser
from pathlib import Path, PurePath
from platform import uname
from typing import TYPE_CHECKING, Union

from .error import InvalidDataTypeError

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    try:
        import pandas as pd
    except ImportError:
        pd = None

    try:
        import numpy as np
    except ImportError:
        np = None


StrPath = Union[str, PurePath]


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
        return None

    wrapper.has_run = False
    return wrapper


@run_once
def open_file_in_browser(file) -> bool:
    path = Path(file)
    url = str(path) if "Microsoft" in uname().release else path.resolve().as_uri()

    return webbrowser.open(url)


def env2bool(var, undefined=False):
    """
    undefined: return value if env var is unset
    """
    var = os.getenv(var, None)
    if var is None:
        return undefined
    return bool(re.search("1|y|yes|true", var, flags=re.IGNORECASE))


def standardize_metric_name(metric_name: str, framework: str) -> str:
    """Map framework-specific format to DVCLive standard.

    Use `{split}/` as prefix in order to separate by subfolders.
    Use `{train|eval}` as split name.
    """
    if framework == "dvclive.fastai":
        metric_name = metric_name.replace("train_", "train/")
        metric_name = metric_name.replace("valid_", "eval/")

    elif framework == "dvclive.huggingface":
        for split in ("train", "eval"):
            metric_name = metric_name.replace(f"{split}_", f"{split}/")

    elif framework == "dvclive.keras":
        if "val_" in metric_name:
            metric_name = metric_name.replace("val_", "eval/")
        else:
            metric_name = f"train/{metric_name}"

    elif framework in ["dvclive.lightning", "dvclive.fabric"]:
        parts = metric_name.split("_")
        split, freq, rest = None, None, None
        if any(parts[0].endswith(split) for split in ["train", "val", "test"]):
            split = parts.pop(0)
            # Only set freq if split was also found.
            # Otherwise we end up conflicting with out internal `step` property.
            if parts[-1] in ["step", "epoch"]:
                freq = parts.pop()
        rest = "_".join(parts)
        parts = [part for part in (split, freq, rest) if part]
        metric_name = "/".join(parts)

    return metric_name


def parse_tsv(path):
    with open(path, encoding="utf-8", newline="") as fd:
        reader = csv.DictReader(fd, delimiter="\t")
        return list(reader)


def parse_json(path):
    with open(path, encoding="utf-8") as fd:
        return json.load(fd)


def parse_metrics(live):
    from .plots import Metric

    metrics_path = Path(live.plots_dir) / Metric.subfolder
    history = {}
    for suffix in Metric.suffixes:
        for scalar_file in metrics_path.rglob(f"*{suffix}"):
            history[str(scalar_file)] = parse_tsv(scalar_file)
    latest = parse_json(live.metrics_file)
    return history, latest


def matplotlib_installed() -> bool:
    try:
        import matplotlib as mpl  # noqa: F401
    except ImportError:
        return False
    return True


def inside_colab() -> bool:
    try:
        from google import colab  # noqa: F401
    except ImportError:
        return False
    return True


def inside_notebook() -> bool:
    if inside_colab():
        return True

    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        return False

    if shell == "ZMQInteractiveShell":
        import IPython

        return IPython.__version__ >= "6.0.0"
    return False


def clean_and_copy_into(src: StrPath, dst: StrPath) -> str:
    Path(dst).mkdir(exist_ok=True)

    basename = os.path.basename(os.path.normpath(src))
    dst_path = Path(os.path.join(dst, basename))

    if dst_path.is_file() or dst_path.is_symlink():
        dst_path.unlink()
    elif dst_path.is_dir():
        shutil.rmtree(dst_path)

    if os.path.isdir(src):
        shutil.copytree(src, dst_path)
    else:
        shutil.copy2(src, dst_path)

    return str(dst_path)


def isinstance_without_import(val, module, name):
    for cls in type(val).mro():
        if (cls.__module__, cls.__name__) == (module, name):
            return True
    return False


def catch_and_warn(exception, logger, on_finally=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception as e:
                logger.warning(f"Error in {func.__name__}: {e}")
            finally:
                if on_finally is not None:
                    on_finally()

        return wrapper

    return decorator


def rel_path(path, dvc_root_path):
    absolute_path = Path(path).absolute()
    return str(Path(os.path.relpath(absolute_path, dvc_root_path)).as_posix())


def read_history(live, metric):
    from dvclive.plots.metric import Metric

    history, _ = parse_metrics(live)
    steps = []
    values = []
    name = os.path.join(live.plots_dir, Metric.subfolder, f"{metric}.tsv")
    for e in history[name]:
        steps.append(int(e["step"]))
        values.append(float(e[metric]))
    return steps, values


def read_latest(live, metric_name):
    _, latest = parse_metrics(live)
    return latest["step"], latest[metric_name]


def convert_datapoints_to_list_of_dicts(
    datapoints: Union[list[dict], "pd.DataFrame", "np.ndarray"],
) -> list[dict]:
    """
    Convert the given datapoints to a list of dictionaries.

    Args:
        datapoints: The input datapoints to be converted.

    Returns:
        A list of dictionaries representing the datapoints.

    Raises:
        TypeError: `datapoints` must be pd.DataFrame, np.ndarray, or List[Dict]
    """
    if isinstance(datapoints, list):
        return datapoints

    if pd and isinstance(datapoints, pd.DataFrame):
        return datapoints.to_dict(orient="records")

    if np and isinstance(datapoints, np.ndarray):
        # This is a structured array
        if datapoints.dtype.names is not None:
            return [dict(zip(datapoints.dtype.names, row)) for row in datapoints]

        # This is a regular array
        return [dict(enumerate(row)) for row in datapoints]

    # Raise an error if the input is not a supported type
    raise InvalidDataTypeError("datapoints", type(datapoints))
