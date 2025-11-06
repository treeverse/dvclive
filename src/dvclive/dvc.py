# ruff: noqa: SLF001
import copy
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from dvclive import env
from dvclive.plots import Image, Metric
from dvclive.serialize import dump_yaml
from dvclive.utils import StrPath, rel_path

if TYPE_CHECKING:
    from dvc.repo import Repo
    from dvc.stage import Stage

logger = logging.getLogger("dvclive")


def _dvc_dir(dirname: StrPath) -> str:
    return os.path.join(dirname, ".dvc")


def _find_dvc_root(root: Optional[StrPath] = None) -> Optional[str]:
    if not root:
        root = os.getcwd()

    root = os.path.realpath(root)

    if not os.path.isdir(root):
        raise NotADirectoryError(f"'{root}'")

    while True:
        if os.path.exists(_dvc_dir(root)):
            return root
        if os.path.ismount(root):
            break
        root = os.path.dirname(root)

    return None


def get_dvc_repo() -> Optional["Repo"]:
    from dvc.exceptions import NotDvcRepoError
    from dvc.repo import Repo
    from dvc.scm import Git, SCMError
    from scmrepo.exceptions import SCMError as GitSCMError

    try:
        return Repo()
    except (NotDvcRepoError, SCMError):
        try:
            return Repo.init(Git().root_dir)
        except GitSCMError:
            return None


def make_dvcyaml(live) -> None:  # noqa: C901
    dvcyaml_dir = Path(live.dvc_file).parent.absolute().as_posix()

    dvcyaml = {}
    if live._params:
        dvcyaml["params"] = [rel_path(live.params_file, dvcyaml_dir)]
    if live._metrics or live.summary:
        dvcyaml["metrics"] = [rel_path(live.metrics_file, dvcyaml_dir)]
    plots: list[Any] = []
    plots_path = Path(live.plots_dir)
    plots_metrics_path = plots_path / Metric.subfolder
    if plots_metrics_path.exists():
        metrics_config = {rel_path(plots_metrics_path, dvcyaml_dir): {"x": "step"}}
        plots.append(metrics_config)
    if live._images:
        images_path = rel_path(plots_path / Image.subfolder, dvcyaml_dir)
        plots.append(images_path)
    if live._plots:
        for plot in live._plots.values():
            plot_path = rel_path(plot.output_path, dvcyaml_dir)
            plots.append({plot_path: plot.plot_config})
    if plots:
        dvcyaml["plots"] = plots

    if live._artifacts:
        dvcyaml["artifacts"] = copy.deepcopy(live._artifacts)
        for artifact in dvcyaml["artifacts"].values():  # type: ignore[attr-defined]
            artifact["path"] = rel_path(artifact["path"], dvcyaml_dir)

    if not os.path.exists(live.dvc_file):
        dump_yaml(dvcyaml, live.dvc_file)
    else:
        update_dvcyaml(live, dvcyaml)


def update_dvcyaml(live, updates):
    from dvc.utils.serialize import modify_yaml

    dvcyaml_dir = os.path.abspath(os.path.dirname(live.dvc_file))
    dvclive_dir = os.path.relpath(live.dir, dvcyaml_dir) + "/"

    def _drop_stale_dvclive_entries(entries):
        non_dvclive = []
        for e in entries:
            if isinstance(e, str):
                if dvclive_dir not in e:
                    non_dvclive.append(e)
            elif isinstance(e, dict) and len(e) == 1:
                if dvclive_dir not in next(iter(e.keys())):
                    non_dvclive.append(e)
            else:
                non_dvclive.append(e)
        return non_dvclive

    def _update_entries(old, new, key):
        keepers = _drop_stale_dvclive_entries(old.get(key, []))
        old[key] = keepers + new.get(key, [])
        if not old[key]:
            del old[key]
        return old

    with modify_yaml(live.dvc_file) as orig:
        orig = _update_entries(orig, updates, "params")  # noqa: PLW2901
        orig = _update_entries(orig, updates, "metrics")  # noqa: PLW2901
        orig = _update_entries(orig, updates, "plots")  # noqa: PLW2901
        old_artifacts = {
            name: meta
            for name, meta in orig.get("artifacts", {}).items()
            if dvclive_dir not in meta.get("path", dvclive_dir)
        }
        orig["artifacts"] = {**old_artifacts, **updates.get("artifacts", {})}
        if not orig["artifacts"]:
            del orig["artifacts"]


def get_exp_name(name, scm, baseline_rev) -> str:
    from dvc.exceptions import InvalidArgumentError
    from dvc.repo.experiments.refs import ExpRefInfo
    from dvc.repo.experiments.utils import (
        check_ref_format,
        gen_random_name,
        get_random_exp_name,
    )

    name = name or os.getenv(env.DVC_EXP_NAME)
    if name and scm and baseline_rev:
        ref = ExpRefInfo(baseline_sha=baseline_rev, name=name)
        if scm.get_ref(str(ref)):
            logger.warning(f"Experiment conflicts with existing experiment '{name}'.")
        else:
            try:
                check_ref_format(scm, ref)
            except InvalidArgumentError as e:
                logger.warning(e)
            else:
                return name
    if scm and baseline_rev:
        return get_random_exp_name(scm, baseline_rev)
    if name:
        return name
    return gen_random_name()


def find_overlapping_stage(dvc_repo: "Repo", path: StrPath) -> Optional["Stage"]:
    abs_path = str(Path(path).absolute())
    for stage in dvc_repo.index.stages:
        for out in stage.outs:
            if str(out.fs_path) in abs_path:
                return stage
    return None


def ensure_dir_is_tracked(directory: str, dvc_repo: "Repo") -> None:
    from pathspec import PathSpec

    dir_spec = PathSpec.from_lines("gitwildmatch", [directory])
    outs_spec = PathSpec.from_lines(
        "gitwildmatch", [str(o) for o in dvc_repo.index.outs]
    )
    paths_to_track = [
        f
        for f in dvc_repo.scm.untracked_files()
        if (dir_spec.match_file(f) and not outs_spec.match_file(f))
    ]
    if paths_to_track:
        dvc_repo.scm.add(paths_to_track)
