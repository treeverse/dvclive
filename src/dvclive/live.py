import builtins
import glob
import json
import logging
import math
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

if TYPE_CHECKING:
    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    import PIL
    from dvc.repo import Repo
    from IPython.display import DisplayHandle

from dvc.exceptions import DvcException
from dvc.utils.studio import get_repo_url, get_subrepo_relpath
from funcy import set_in
from ruamel.yaml.representer import RepresenterError

from . import env
from .dvc import (
    ensure_dir_is_tracked,
    find_overlapping_stage,
    get_dvc_repo,
    get_exp_name,
    make_dvcyaml,
)
from .error import (
    InvalidDataTypeError,
    InvalidDvcyamlError,
    InvalidImageNameError,
    InvalidParameterTypeError,
    InvalidPlotTypeError,
    InvalidReportModeError,
)
from .monitor_system import _SystemMonitor
from .plots import PLOT_TYPES, SKLEARN_PLOTS, CustomPlot, Image, Metric, NumpyEncoder
from .report import BLANK_NOTEBOOK_REPORT, make_report
from .serialize import dump_json, dump_yaml, load_yaml
from .studio import get_dvc_studio_config, post_to_studio
from .utils import (
    StrPath,
    catch_and_warn,
    clean_and_copy_into,
    convert_datapoints_to_list_of_dicts,
    env2bool,
    inside_notebook,
    matplotlib_installed,
    open_file_in_browser,
    parse_metrics,
)
from .vscode import (
    cleanup_dvclive_step_completed,
    mark_dvclive_only_ended,
    mark_dvclive_only_started,
    mark_dvclive_step_completed,
)

logger = logging.getLogger("dvclive")
logger.setLevel(os.getenv(env.DVCLIVE_LOGLEVEL, "WARNING").upper())
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

ParamLike = Union[int, float, str, bool, list["ParamLike"], dict[str, "ParamLike"]]

NULL_SHA: str = "0" * 40


class Live:
    def __init__(
        self,
        dir: str = "dvclive",  # noqa: A002
        resume: bool = False,
        report: Optional[Literal["md", "notebook", "html"]] = None,
        save_dvc_exp: bool = True,
        dvcyaml: Union[str, os.PathLike, bool, None] = "dvc.yaml",
        cache_images: bool = False,
        exp_name: Optional[str] = None,
        exp_message: Optional[str] = None,
        monitor_system: bool = False,
    ):
        """
        Initializes a DVCLive logger. A `Live()` instance is required in order to log
        machine learning parameters, metrics and other metadata.
        Warning: `Live()` will remove all existing DVCLive related files under dir
        unless `resume=True`.

        Args:
            dir (str | Path): where to save DVCLive's outputs. Defaults to `"dvclive"`.
            resume (bool): if `True`, DVCLive will try to read the previous step from
                the metrics_file and start from that point. Defaults to `False`.
            report ("html", "md", "notebook", None): any of `"html"`, `"notebook"`,
                `"md"` or `None`. See `Live.make_report()`. Defaults to None.
            save_dvc_exp (bool): if `True`, DVCLive will create a new DVC experiment as
                part of `Live.end()`. Defaults to `True`. If you are using DVCLive
                inside a DVC Pipeline and running with `dvc exp run`, the option will be
                ignored.
            dvcyaml (str | Path | None): where to write dvc.yaml file, which adds DVC
                configuration for metrics, plots, and parameters as part of
                `Live.next_step()` and `Live.end()`. If `None`, no dvc.yaml file is
                written. Defaults to `"dvc.yaml"`. See `Live.make_dvcyaml()`.
                If a string or Path like `"subdir/dvc.yaml"`, DVCLive will write the
                configuration to that path (file must be named "dvc.yaml").
                If `False`, DVCLive will not write to "dvc.yaml" (useful if you are
                tracking DVCLive metrics, plots, and parameters independently and
                want to avoid duplication).
            cache_images (bool): if `True`, DVCLive will cache any images logged with
                `Live.log_image()` as part of `Live.end()`. Defaults to `False`.
                If running a DVC pipeline, `cache_images` will be ignored, and you
                should instead cache images as pipeline outputs.
            exp_name (str | None): if not `None`, and `save_dvc_exp` is `True`, the
                provided string will be passed to `dvc exp save --name`.
                If DVCLive is used inside `dvc exp run`, the option will be ignored, use
                `dvc exp run --name` instead.
            exp_message (str | None): if not `None`, and `save_dvc_exp` is `True`, the
                provided string will be passed to `dvc exp save --message`.
                If DVCLive is used inside `dvc exp run`, the option will be ignored, use
                `dvc exp run --message` instead.
            monitor_system (bool): if `True`, DVCLive will monitor GPU, CPU, ram, and
                disk usage. Defaults to `False`.
        """
        self.summary: dict[str, Any] = {}

        self._dir: str = dir
        self._resume: bool = resume or env2bool(env.DVCLIVE_RESUME)
        self._save_dvc_exp: bool = save_dvc_exp
        self._step: Optional[int] = None
        self._metrics: dict[str, Any] = {}
        self._images: dict[str, Image] = {}
        self._params: dict[str, Any] = {}
        self._plots: dict[str, Any] = {}
        self._artifacts: dict[str, dict] = {}
        self._inside_with = False
        self._dvcyaml = dvcyaml
        self._cache_images = cache_images

        self._report_mode: Optional[str] = report
        self._report_notebook: Optional[DisplayHandle] = None
        self._init_report()

        self._baseline_rev: str = os.getenv(env.DVC_EXP_BASELINE_REV, NULL_SHA)
        self._exp_name: Optional[str] = exp_name or os.getenv(env.DVC_EXP_NAME)
        self._exp_message: Optional[str] = exp_message
        self._subdir: Optional[str] = None
        self._repo_url: Optional[str] = None
        self._experiment_rev: Optional[str] = None
        self._inside_dvc_exp: bool = False
        self._inside_dvc_pipeline: bool = False
        self._dvc_repo: Optional[Repo] = None
        self._include_untracked: list[str] = []
        if env2bool(env.DVCLIVE_TEST):
            self._init_test()
        else:
            self._init_dvc()

        os.makedirs(self.dir, exist_ok=True)

        if self._resume:
            self._init_resume()
        else:
            self._init_cleanup()

        self._latest_studio_step: int = self.step if resume else -1
        self._studio_events_to_skip: set[str] = set()
        self._dvc_studio_config: dict[str, Any] = {}
        self._num_points_sent_to_studio: dict[str, int] = {}
        self._studio_queue = None
        self._init_studio()

        self._system_monitor: Optional[_SystemMonitor] = None  # Monitoring thread
        if monitor_system:
            self.monitor_system()

    def _init_resume(self):
        self._read_params()
        self.summary = self.read_latest()
        self._step = self.read_step()
        if self._step != 0:
            logger.info(f"Resuming from step {self._step}")
            self._step += 1
        logger.debug(f"{self._step=}")

    def _init_cleanup(self):
        for plot_type in PLOT_TYPES:
            shutil.rmtree(
                Path(self.plots_dir) / plot_type.subfolder, ignore_errors=True
            )

        for f in (
            self.metrics_file,
            self.params_file,
            os.path.join(self.dir, "report.html"),
            os.path.join(self.dir, "report.md"),
        ):
            if f and os.path.exists(f):
                os.remove(f)

        for dvc_file in glob.glob(os.path.join(self.dir, "**dvc.yaml")):
            os.remove(dvc_file)

    @catch_and_warn(DvcException, logger)
    def _init_dvc(self):  # noqa: C901
        from dvc.scm import NoSCM

        if os.getenv(env.DVC_ROOT, None):
            self._inside_dvc_pipeline = True
            self._init_dvc_pipeline()
        self._dvc_repo = get_dvc_repo()

        scm = self._dvc_repo.scm if self._dvc_repo else None
        if isinstance(scm, NoSCM):
            scm = None
        if scm:
            self._baseline_rev = scm.get_rev()
        self._exp_name = get_exp_name(self._exp_name, scm, self._baseline_rev)
        logger.info(f"Logging to experiment '{self._exp_name}'")

        dvc_logger = logging.getLogger("dvc")
        dvc_logger.setLevel(os.getenv(env.DVCLIVE_LOGLEVEL, "WARNING").upper())

        self._dvc_file = self._init_dvc_file()

        if not scm:
            if self._save_dvc_exp:
                logger.warning(
                    "Can't save experiment without a Git Repo."
                    "\nCreate a Git repo (`git init`) and commit (`git commit`)."
                )
                self._save_dvc_exp = False
            return
        if scm.no_commits:
            if self._save_dvc_exp:
                logger.warning(
                    "Can't save experiment to an empty Git Repo."
                    "\nAdd a commit (`git commit`) to save experiments."
                )
                self._save_dvc_exp = False
            return

        if self._dvcyaml and (
            stage := find_overlapping_stage(self._dvc_repo, self.dvc_file)
        ):
            logger.warning(
                f"'{self.dvc_file}' is in outputs of stage '{stage.addressing}'."
                "\nRemove it from outputs to make DVCLive work as expected."
            )

        if self._inside_dvc_pipeline:
            return

        self._subdir = get_subrepo_relpath(self._dvc_repo)
        self._repo_url = get_repo_url(self._dvc_repo)

        if self._save_dvc_exp:
            mark_dvclive_only_started(self._exp_name)
            self._include_untracked.append(self.dir)

    def _init_dvc_file(self) -> str:
        if self._dvcyaml is None:
            return "dvc.yaml"
        if isinstance(self._dvcyaml, bool):
            return "dvc.yaml"

        self._dvcyaml = os.fspath(self._dvcyaml)
        if (
            isinstance(self._dvcyaml, str)
            and os.path.basename(self._dvcyaml) == "dvc.yaml"
        ):
            return self._dvcyaml

        raise InvalidDvcyamlError

    def _init_dvc_pipeline(self):
        if os.getenv(env.DVC_EXP_BASELINE_REV, None):
            # `dvc exp` execution
            self._inside_dvc_exp = True
            if self._save_dvc_exp:
                logger.info("Ignoring `save_dvc_exp` because `dvc exp run` is running")
        # `dvc repro` execution
        elif self._save_dvc_exp:
            logger.warning(
                "Ignoring `save_dvc_exp` because `dvc repro` is running."
                "\nUse `dvc exp run` to save experiment."
            )
        self._save_dvc_exp = False

    def _init_studio(self):
        self._dvc_studio_config = get_dvc_studio_config(self)
        if not self._dvc_studio_config:
            logger.debug("Skipping `studio` report.")
            self._studio_events_to_skip.add("start")
            self._studio_events_to_skip.add("data")
            self._studio_events_to_skip.add("done")
        elif self._inside_dvc_exp:
            logger.debug("Skipping `studio` report `start` and `done` events.")
            self._studio_events_to_skip.add("start")
            self._studio_events_to_skip.add("done")
        else:
            post_to_studio(self, "start")

    def _init_report(self):
        if self._report_mode not in {None, "html", "notebook", "md"}:
            raise InvalidReportModeError(self._report_mode)
        if self._report_mode == "notebook":
            if inside_notebook():
                from IPython.display import Markdown, display

                self._report_mode = "notebook"
                self._report_notebook = display(
                    Markdown(BLANK_NOTEBOOK_REPORT), display_id=True
                )
            else:
                logger.warning(
                    "Report mode 'notebook' requires to be"
                    " inside a notebook. Disabling report."
                )
                self._report_mode = None
        if self._report_mode in ("notebook", "md") and not matplotlib_installed():
            logger.warning(
                f"Report mode '{self._report_mode}' requires 'matplotlib'"
                " to be installed. Disabling report."
            )
            self._report_mode = None
        logger.debug(f"{self._report_mode=}")

    def _init_test(self):
        """
        Enables a test mode that writes to temporary paths and doesn't depend on the
        repository.

        This is needed to run integration tests in external libraries, such as
        HuggingFace Accelerate.
        """
        with tempfile.TemporaryDirectory() as dirpath:
            self._dir = os.path.join(dirpath, self._dir)
            self._dvcyaml = os.fspath(self._dvcyaml)
            if isinstance(self._dvcyaml, str):
                self._dvc_file = os.path.join(dirpath, self._dvcyaml)
            self._save_dvc_exp = False
            logger.warning(
                "DVCLive testing mode enabled."
                f"Repo will be ignored and output will be written to {dirpath}."
            )

    @property
    def dir(self) -> str:
        """Location of the directory to store outputs."""
        return self._dir

    @property
    def params_file(self) -> str:
        return os.path.join(self.dir, "params.yaml")

    @property
    def metrics_file(self) -> str:
        return os.path.join(self.dir, "metrics.json")

    @property
    def dvc_file(self) -> str:
        """Path for dvc.yaml file."""
        return self._dvc_file

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.dir, "plots")

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.dir, "artifacts")

    @property
    def report_file(self) -> Optional[str]:
        if self._report_mode in ("html", "md"):
            suffix = self._report_mode
            return os.path.join(self.dir, f"report.{suffix}")
        return None

    @property
    def step(self) -> int:
        return self._step or 0

    @step.setter
    def step(self, value: int) -> None:
        self._step = value
        logger.debug(f"Step: {self.step}")

    def monitor_system(
        self,
        interval: float = 0.05,  # seconds
        num_samples: int = 20,
        directories_to_monitor: Optional[dict[str, str]] = None,
    ) -> None:
        """Monitor GPU, CPU, ram, and disk resources and log them to DVC Live.

        Args:
            interval (float): the time interval between samples in seconds. To keep the
                sampling interval small, the maximum value allowed is 0.1 seconds.
                Default to 0.05.
            num_samples (int): the number of samples to collect before the aggregation.
                The value should be between 1 and 30 samples. Default to 20.
            directories_to_monitor (Optional[Dict[str, str]]): a dictionary with the
                information about which directories to monitor. The `key` would be the
                name of the metric and the `value` is the path to the directory.
                The metric tracked concerns the partition that contains the directory.
                Default to `{"main": "/"}`.

        Raises:
            ValueError: if the keys in `directories_to_monitor` contains invalid
                characters as defined by `os.path.normpath`.
        """
        if directories_to_monitor is None:
            directories_to_monitor = {"main": "/"}

        if self._system_monitor is not None:
            self._system_monitor.end()

        self._system_monitor = _SystemMonitor(
            live=self,
            interval=interval,
            num_samples=num_samples,
            directories_to_monitor=directories_to_monitor,
        )

    def sync(self):
        self.make_summary()

        if self._dvcyaml:
            self.make_dvcyaml()

        self.make_report()

        self.post_data_to_studio()

    def next_step(self):
        """
        Signals that the current iteration has ended and increases step value by one.
        DVCLive uses `step` to track the history of the metrics logged with
        `Live.log_metric()`.
        You can use `Live.next_step()` to increase the step by one. In addition to
        increasing the `step` number, it will call `Live.make_report()`,
        `Live.make_dvcyaml()`, and `Live.make_summary()` by default.
        """
        if self._step is None:
            self._step = 0

        self.sync()
        mark_dvclive_step_completed(self.step)
        self.step += 1

    def log_metric(
        self,
        name: str,
        val: Union[float, str],
        timestamp: bool = False,
        plot: bool = True,
    ):
        """
        On each `Live.log_metric(name, val)` call `DVCLive` will create a metrics
        history file in `{Live.plots_dir}/metrics/{name}.tsv`. Each subsequent call to
        `Live.log_metric(name, val)` will add a new row to
        `{Live.plots_dir}/metrics/{name}.tsv`. In addition, `DVCLive` will store the
        latest value logged in `Live.summary`, so it can be serialized with calls to
        `live.make_summary()`, `live.next_step()` or when exiting the `Live` context
        block.

        Args:
            name (str): name of the metric being logged.
            val (int | float | str): the value to be logged.
            timestamp (bool): whether to automatically log timestamp in the metrics
                history file.
            plot (bool): whether to add the metric value to the metrics history file for
                plotting. If `False`, the metric will only be saved to the metrics
                summary.

        Raises:
            `InvalidDataTypeError`: thrown if the provided `val` does not have a
                supported type.
        """
        if not Metric.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if not isinstance(val, str) and (math.isnan(val) or math.isinf(val)):
            val = str(val)

        if name in self._metrics:
            metric = self._metrics[name]
        else:
            metric = Metric(name, self.plots_dir)
            self._metrics[name] = metric

        metric.step = self.step
        if plot:
            metric.dump(val, timestamp=timestamp)

        self.summary = set_in(self.summary, metric.summary_keys, val)
        logger.debug(f"Logged {name}: {val}")

    def log_image(
        self,
        name: str,
        val: "Union[np.ndarray, mpl.figure.Figure, PIL.Image.Image, StrPath]",
    ):
        """
        Saves the given image `val` to the output file `name`.

        Supported values for val are:
        - A valid NumPy array (convertible to an image via `PIL.Image.fromarray`)
        - A `matplotlib.figure.Figure` instance
        - A `PIL.Image` instance
        - A path to an image file (`str` or `Path`). It should be in a format that is
        readable by `PIL.Image.open()`

        The images will be saved in `{Live.plots_dir}/images/{name}`. When using
        `Live(cache_images=True)`, the images directory will also be cached as part of
        `Live.end()`. In that case, a `.dvc` file will be saved to track it, and the
        directory will be added to a `.gitignore` file to prevent Git tracking.

        By default the images will be overwritten on each step. However, you can log
        images using the following pattern
        `live.log_image(f"folder/{live.step}.png", img)`.
        In `DVC Studio` and the `DVC Extension for VSCode`, folders following this
        pattern will be rendered using an image slider.

        Args:
            name (str): name of the image file that this command will output
            val (np.ndarray | matplotlib.figure.Figure | PIL.Image | StrPath):
                image to be saved. See the list of supported values in the description.

        Raises:
            `InvalidDataTypeError`: thrown if the provided `val` does not have a
                supported type.
        """
        if not Image.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        # If we're given a path, try loading the image first. This might error out.
        if isinstance(val, (str, PurePath)):
            from PIL import Image as ImagePIL

            suffix = Path(val).suffix
            if not Path(name).suffix and suffix in Image.suffixes:
                name = f"{name}{suffix}"

            val = ImagePIL.open(val)

        # See if the image name is valid
        if Path(name).suffix not in Image.suffixes:
            raise InvalidImageNameError(name)

        if name in self._images:
            image = self._images[name]
        else:
            image = Image(name, self.plots_dir)
            self._images[name] = image

        image.step = self.step
        image.dump(val)
        logger.debug(f"Logged {name}: {val}")

    def log_plot(
        self,
        name: str,
        datapoints: "Union[pd.DataFrame, np.ndarray, list[dict]]",
        x: str,
        y: Union[str, list[str]],
        template: Optional[str] = "linear",
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ):
        """
        The method will dump the provided datapoints to
        `{Live.dir}/plots/custom/{name}.json`and store the provided properties to be
        included in the plots section written by `Live.make_dvcyaml()`. The plot can be
        rendered with `DVC CLI`, `VSCode Extension` or `DVC Studio`.

        Args:
            name (StrPath): name of the output file.
            datapoints (pd.DataFrame | np.ndarray | List[Dict]): Pandas DataFrame, Numpy
                Array or List of dictionaries containing the data for the plot.
            x (str): name of the key (present in the dictionaries) to use as the x axis.
            y (str | list[str]): name of the key or keys (present in the
                dictionaries) to use the y axis.
            template (str): name of the `DVC plots template` to use. Defaults to
                `"linear"`.
            title (str): title to be displayed. Defaults to
                `"{Live.dir}/plots/custom/{name}.json"`.
            x_label (str): label for the x axis. Defaults to the name passed as `x`.
            y_label (str): label for the y axis. Defaults to the name passed as `y`.

        Raises:
            `InvalidDataTypeError`: thrown if the provided `datapoints` does not have a
                supported type.
        """
        # Convert the given datapoints to List[Dict]
        datapoints = convert_datapoints_to_list_of_dicts(datapoints=datapoints)

        if not CustomPlot.could_log(datapoints):
            raise InvalidDataTypeError(name, type(datapoints))

        if name in self._plots:
            plot = self._plots[name]
        else:
            plot = CustomPlot(
                name,
                self.plots_dir,
                x=x,
                y=y,
                template=template,
                title=title,
                x_label=x_label,
                y_label=y_label,
            )
            self._plots[name] = plot

        plot.step = self.step
        plot.dump(datapoints)
        logger.debug(f"Logged {name}")

    def log_sklearn_plot(
        self,
        kind: str,
        labels: "Union[list, np.ndarray]",
        predictions: "Union[list, tuple, np.ndarray]",
        name: Optional[str] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        normalized: Optional[bool] = None,
        **kwargs,
    ):
        """
        Generates a scikit learn plot and saves the data in
        `{Live.dir}/plots/sklearn/{name}.json`. The method will compute and dump the
        `kind` plot to `{Live.dir}/plots/sklearn/{name}` in a format compatible with
        dvc plots. It will also store the provided properties to be included in the
        plots section written by `Live.make_dvcyaml()`.

        Args:
            kind ("calibration" | "confusion_matrix" | "det" | "precision_recall" |
                "roc"): a supported plot type.
            labels (List | np.ndarray): array of ground truth labels.
            predictions (List | np.ndarray): array of predicted labels (for
                `"confusion_matrix"`) or predicted probabilities (for other plots).
            name (str): optional name of the output file. If not provided, `kind` will
                be used as name.
            title (str): optional title to be displayed.
            x_label (str): optional label for the x axis.
            y_label (str): optional label for the y axis.
            normalized (bool): optional, `confusion_matrix` with values normalized to
                `<0, 1>` range.
            kwargs: additional arguments to tune the result. Arguments are passed to the
                scikit-learn function (e.g. `drop_intermediate=True` for the `"roc"`
                type).
        Raises:
            InvalidPlotTypeError: thrown if the provided `kind` does not correspond to
                any of the supported plots.
        """
        val = (labels, predictions)

        plot_config = {
            k: v
            for k, v in {
                "title": title,
                "x_label": x_label,
                "y_label": y_label,
                "normalized": normalized,
            }.items()
            if v is not None
        }

        name = name or kind
        if name in self._plots:
            plot = self._plots[name]
        elif kind in SKLEARN_PLOTS and SKLEARN_PLOTS[kind].could_log(val):
            plot = SKLEARN_PLOTS[kind](name, self.plots_dir, **plot_config)
            self._plots[plot.name] = plot
        else:
            raise InvalidPlotTypeError(name)

        plot.step = self.step
        plot.dump(val, **kwargs)
        logger.debug(f"Logged {name}")

    def _read_params(self):
        if os.path.isfile(self.params_file):
            params = load_yaml(self.params_file)
            self._params.update(params)

    def _dump_params(self):
        try:
            dump_yaml(self._params, self.params_file)
        except RepresenterError as exc:
            raise InvalidParameterTypeError(exc.args[0]) from exc

    def log_params(self, params: dict[str, ParamLike]):
        """
        On each `Live.log_params(params)` call, DVCLive will write keys/values pairs in
        the params dict to `{Live.dir}/params.yaml`:

        Also see `Live.log_param()`.

        Args:
            params (Dict[str, ParamLike]): dictionary with name/value pairs of
                parameters to be logged.

        Raises:
            `InvalidParameterTypeError`: thrown if the parameter value is not among
                supported types.
        """
        self._params.update(params)
        self._dump_params()
        logger.debug(f"Logged {params} parameters to {self.params_file}")

    def log_param(self, name: str, val: ParamLike):
        """
        On each `Live.log_param(name, val)` call, DVCLive will write the name parameter
        to `{Live.dir}/params.yaml` with the corresponding `val`.

        Also see `Live.log_params()`.

        Args:
            name (str): name of the parameter being logged.
            val (ParamLike): the value to be logged.

        Raises:
            `InvalidParameterTypeError`: thrown if the parameter value is not among
                supported types.
        """
        self.log_params({name: val})

    def log_artifact(
        self,
        path: StrPath,
        type: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        desc: Optional[str] = None,
        labels: Optional[list[str]] = None,
        meta: Optional[dict[str, Any]] = None,
        copy: bool = False,
        cache: bool = True,
    ):
        """
        Tracks an existing directory or file with DVC.

        Log path, saving its contents to DVC storage. Also annotate with any included
        metadata fields (for example, to be consumed in the model registry or automation
        scenarios).
        If `cache=True` (which is the default), uses `dvc add` to track path with DVC,
        saving it to the DVC cache and generating a `{path}.dvc` file that acts as a
        pointer to the cached data.
        If you include any of the optional metadata fields (type, name, desc, labels,
        meta), it will add an artifact and all the metadata passed as arguments to the
        corresponding `dvc.yaml` (unless `dvcyaml=None`). Passing `type="model"` will
        include it in the model registry.

        Args:
            path (StrPath): an existing directory or file.
            type (Optional[str]): an optional type of the artifact. Common types are
                `"model"` or `"dataset"`.
            name (Optional[str]): an optional custom name of an artifact.
                If not provided the `path` stem (last part of the path without the
                file extension) will be used as the artifact name.
            desc (Optional[str]): an optional description of an artifact.
            labels (Optional[List[str]]): optional labels describing the artifact.
            meta (Optional[Dict[str, Any]]): optional metainformation in `key: value`
                format.
            copy (bool): copy a directory or file at path into the `dvclive/artifacts`
                location (default) before tracking it. The new path is used instead of
                the original one to track the artifact. Useful if you don't want to
                track the original path in your repo (for example, it is outside the
                repo or in a Git-ignored directory).
            cache (bool): cache the files with DVC to track them outside of Git.
                Defaults to `True`, but set to `False` if you want to annotate metadata
                about the artifact without storing a copy in the DVC cache.
                If running a DVC pipeline, `cache` will be ignored, and you should
                instead cache artifacts as pipeline outputs.

        Raises:
            `InvalidDataTypeError`: thrown if the provided `path` does not have a
                supported type.
        """
        if not isinstance(path, (str, PurePath)):
            raise InvalidDataTypeError(path, builtins.type(path))

        if self._dvc_repo is not None:
            from gto.constants import assert_name_is_valid
            from gto.exceptions import ValidationError

            if copy:
                path = clean_and_copy_into(path, self.artifacts_dir)

            if cache:
                self.cache(path)

            if any((type, name, desc, labels, meta)):
                name = name or Path(path).stem
                try:
                    assert_name_is_valid(name)
                    self._artifacts[name] = {
                        k: v
                        for k, v in locals().items()
                        if k in ("path", "type", "desc", "labels", "meta")
                        and v is not None
                    }
                except ValidationError:
                    logger.warning(
                        "Can't use '%s' as artifact name (ID)."
                        " It will not be included in the `artifacts` section.",
                        name,
                    )
        else:
            logger.warning(
                "A DVC repo is required to log artifacts. "
                f"Skipping `log_artifact({path})`."
            )

    @catch_and_warn(DvcException, logger)
    def cache(self, path):
        if self._inside_dvc_pipeline:
            existing_stage = find_overlapping_stage(self._dvc_repo, path)

            if existing_stage:
                if existing_stage.cmd:
                    logger.info(
                        f"Skipping `dvc add {path}` because it is already being"
                        " tracked automatically as an output of the DVC pipeline."
                    )
                    return  # skip caching
                logger.warning(
                    f"To track '{path}' automatically in the DVC pipeline:"
                    f"\n1. Run `dvc remove {existing_stage.addressing}` "
                    "to stop tracking it outside the pipeline."
                    "\n2. Add it as an output of the pipeline stage."
                )
            else:
                logger.warning(
                    f"To track '{path}' automatically in the DVC pipeline, "
                    "add it as an output of the pipeline stage."
                )

        stage = self._dvc_repo.add(str(path))

        dvc_file = stage[0].addressing

        if self._save_dvc_exp:
            self._include_untracked.append(dvc_file)
            self._include_untracked.append(str(Path(dvc_file).parent / ".gitignore"))

    def make_summary(self):
        """
        Serializes a summary of the logged metrics (`Live.summary`) to
        `Live.metrics_file`.

        The `Live.summary` object will contain the latest value of each metric logged
        with `Live.log_metric()`. It can be also modified manually.

        `Live.next_step()` and `Live.end()` will call `Live.make_summary()` internally,
        so you don't need to call both.

        The summary is usable by `dvc metrics`.
        """
        if self._step is not None:
            self.summary["step"] = self.step
        dump_json(self.summary, self.metrics_file, cls=NumpyEncoder)

    def make_report(self):
        """
        Generates a report from the logged data.

        `Live.next_step()` and `Live.end()` will call `Live.make_report()` internally,
        so you don't need to call both.

        On each call, DVCLive will collect all the data logged in `{Live.dir}`, generate
        a report and save it in `{Live.dir}/report.{format}`. The format can be HTML
        or Markdown depending on the value of the `report` argument passed to `Live()`.
        """
        if self._report_mode is not None:
            make_report(self)
            if self._report_mode == "html" and env2bool(env.DVCLIVE_OPEN):
                open_file_in_browser(self.report_file)

    @catch_and_warn(DvcException, logger)
    def make_dvcyaml(self):
        """
        Writes DVC configuration for metrics, plots, and parameters to `Live.dvc_file`.

        Creates `dvc.yaml`, which describes and configures metrics, plots, and
        parameters. DVC tools use this file to show reports and experiments tables.
        `Live.next_step()` and `Live.end()` will call `Live.make_dvcyaml()` internally,
        so you don't need to call both (unless `dvcyaml=None`).
        """
        make_dvcyaml(self)

    def _get_live_data(self) -> Optional[dict[str, Any]]:
        params = load_yaml(self.params_file) if os.path.isfile(self.params_file) else {}
        plots, metrics = parse_metrics(self)

        # Plots can grow large, we don't want to keep in memory data
        # that we 100% sent already
        plots_to_send = {}
        plots_start_idx = {}
        for name, plot in plots.items():
            num_points_sent = self._num_points_sent_to_studio.get(name, 0)
            plots_to_send[name] = plot[num_points_sent:]
            plots_start_idx[name] = num_points_sent

        return {
            "params": params,
            "plots": plots_to_send,
            "plots_start_idx": plots_start_idx,
            "metrics": metrics,
            "images": list(self._images.values()),
            "step": self.step,
        }

    def post_data_to_studio(self):
        if not self._studio_queue:
            self._studio_queue = queue.Queue()

            def worker():
                error_occurred = False
                while True:
                    item, data = self._studio_queue.get()
                    try:
                        if not error_occurred:
                            post_to_studio(item, "data", data)
                    except Exception:
                        logger.exception("Failed to post data to studio")
                        error_occurred = True
                    finally:
                        self._studio_queue.task_done()

            threading.Thread(target=worker, daemon=True).start()

        self._studio_queue.put((self, self._get_live_data()))

    def _wait_for_studio_updates_posted(self):
        if self._studio_queue:
            logger.debug("Waiting for studio updates to be posted")
            self._studio_queue.join()

    def end(self):
        """
        Signals that the current experiment has ended.
        `Live.end()` gets automatically called when exiting the context manager. It is
        also called when the training ends for each of the supported ML Frameworks

        By default, `Live.end()` will call `Live.make_summary()`, `Live.make_dvcyaml()`,
        and `Live.make_report()`.

        If `save_dvc_exp=True`, it will save a new DVC experiment and write a `dvc.yaml`
        file configuring what DVC will show for logged plots, metrics, and parameters.
        """
        if self._inside_with:
            # Prevent `live.end` calls inside context manager
            return

        if self._images and self._cache_images:
            images_path = Path(self.plots_dir) / Image.subfolder
            self.cache(images_path)

        # If next_step called before end, don't want to update step number
        if "step" in self.summary:
            self.step = self.summary["step"]

        # Kill threads that monitor the system metrics
        if self._system_monitor is not None:
            self._system_monitor.end()

        self.sync()

        if self._inside_dvc_exp and self._dvc_repo:
            catch_and_warn(DvcException, logger)(ensure_dir_is_tracked)(
                self.dir, self._dvc_repo
            )
            if self._dvcyaml:
                catch_and_warn(DvcException, logger)(self._dvc_repo.scm.add)(
                    self.dvc_file
                )

        self.save_dvc_exp()

        self._wait_for_studio_updates_posted()

        # Mark experiment as done
        post_to_studio(self, "done")

        cleanup_dvclive_step_completed()

    def read_step(self):
        latest = self.read_latest()
        return latest.get("step", 0)

    def read_latest(self):
        if Path(self.metrics_file).exists():
            with open(self.metrics_file, encoding="utf-8") as fobj:
                return json.load(fobj)
        return {}

    def __enter__(self):
        self._inside_with = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._inside_with = False
        self.end()

    @catch_and_warn(DvcException, logger, mark_dvclive_only_ended)
    def save_dvc_exp(self):
        if self._save_dvc_exp:
            if self._dvcyaml:
                self._include_untracked.append(self.dvc_file)
            self._experiment_rev = self._dvc_repo.experiments.save(
                name=self._exp_name,
                include_untracked=self._include_untracked,
                force=True,
                message=self._exp_message,
            )
