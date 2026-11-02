import sys

import pytest
from dvc_studio_client.env import DVC_STUDIO_TOKEN, DVC_STUDIO_URL, STUDIO_REPO_URL

from dvclive.utils import rel_path


@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def mocked_dvc_repo(tmp_dir, mocker):
    _dvc_repo = mocker.MagicMock()
    _dvc_repo.index.stages = []
    _dvc_repo.scm.get_rev.return_value = "f" * 40
    _dvc_repo.scm.get_ref.return_value = None
    _dvc_repo.scm.no_commits = False
    _dvc_repo.experiments.save.return_value = "e" * 40
    _dvc_repo.root_dir = _dvc_repo.scm.root_dir = tmp_dir
    _dvc_repo.fs.relpath = rel_path
    _dvc_repo.config = {}
    mocker.patch("dvclive.live.get_dvc_repo", return_value=_dvc_repo)
    return _dvc_repo


@pytest.fixture
def mocked_dvc_subrepo(tmp_dir, mocker, mocked_dvc_repo):
    mocked_dvc_repo.root_dir = tmp_dir / "subdir"
    return mocked_dvc_repo


@pytest.fixture
def dvc_repo(tmp_dir):
    from dvc.repo import Repo
    from scmrepo.git import Git

    Git.init(tmp_dir)
    repo = Repo.init(tmp_dir)
    # adding "." with dulwich would also stage the files inside `.git`
    repo.scm.add_commit([".dvc", ".dvcignore"], "init")
    return repo


@pytest.fixture(autouse=True)
def _capture_wrap():
    # https://github.com/pytest-dev/pytest/issues/5502#issuecomment-678368525
    sys.stderr.close = lambda *args: None
    sys.stdout.close = lambda *args: None


@pytest.fixture(autouse=True)
def _mocked_webbrowser_open(mocker):
    mocker.patch("webbrowser.open")


@pytest.fixture(autouse=True)
def _mocked_ci(monkeypatch):
    monkeypatch.setenv("CI", "false")


@pytest.fixture
def mocked_studio_post(mocker, monkeypatch):
    valid_response = mocker.MagicMock()
    valid_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    monkeypatch.setenv(DVC_STUDIO_URL, "https://0.0.0.0")
    monkeypatch.setenv(STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(DVC_STUDIO_TOKEN, "STUDIO_TOKEN")
    return mocked_post, valid_response
