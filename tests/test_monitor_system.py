import time
from pathlib import Path

import dpath
import pytest
from pytest_voluptuous import S

from dvclive import Live
from dvclive.monitor_system import (
    GIGABYTES_DIVIDER,
    METRIC_CPU_COUNT,
    METRIC_CPU_PARALLELIZATION_PERCENT,
    METRIC_CPU_USAGE_PERCENT,
    METRIC_DISK_TOTAL_GB,
    METRIC_DISK_USAGE_GB,
    METRIC_DISK_USAGE_PERCENT,
    METRIC_GPU_COUNT,
    METRIC_GPU_USAGE_PERCENT,
    METRIC_RAM_TOTAL_GB,
    METRIC_RAM_USAGE_GB,
    METRIC_RAM_USAGE_PERCENT,
    METRIC_VRAM_TOTAL_GB,
    METRIC_VRAM_USAGE_GB,
    METRIC_VRAM_USAGE_PERCENT,
    _SystemMonitor,
)
from dvclive.utils import parse_metrics


def mock_psutil_cpu(mocker):
    mocker.patch(
        "dvclive.monitor_system.psutil.cpu_percent",
        return_value=[10, 10, 10, 40, 50, 60],
    )
    mocker.patch("dvclive.monitor_system.psutil.cpu_count", return_value=6)


def mock_psutil_ram(mocker):
    mocked_ram = mocker.MagicMock()
    mocked_ram.percent = 50
    mocked_ram.used = 2 * GIGABYTES_DIVIDER
    mocked_ram.total = 4 * GIGABYTES_DIVIDER
    mocker.patch(
        "dvclive.monitor_system.psutil.virtual_memory", return_value=mocked_ram
    )


def mock_psutil_disk(mocker):
    mocked_disk = mocker.MagicMock()
    mocked_disk.percent = 50
    mocked_disk.used = 16 * GIGABYTES_DIVIDER
    mocked_disk.total = 32 * GIGABYTES_DIVIDER
    mocker.patch("dvclive.monitor_system.psutil.disk_usage", return_value=mocked_disk)


def mock_psutil_disk_with_oserror(mocker):
    mocked_disk = mocker.MagicMock()
    mocked_disk.percent = 50
    mocked_disk.used = 16 * GIGABYTES_DIVIDER
    mocked_disk.total = 32 * GIGABYTES_DIVIDER
    mocker.patch(
        "dvclive.monitor_system.psutil.disk_usage",
        side_effect=[
            mocked_disk,
            OSError,
            mocked_disk,
            OSError,
        ],
    )


def mock_pynvml(mocker, num_gpus=2):
    prefix = "dvclive.monitor_system"
    mocker.patch(f"{prefix}.GPU_AVAILABLE", bool(num_gpus))
    mocker.patch(f"{prefix}.nvmlDeviceGetCount", return_value=num_gpus)
    mocker.patch(f"{prefix}.nvmlInit", return_value=None)
    mocker.patch(f"{prefix}.nvmlShutdown", return_value=None)
    mocker.patch(f"{prefix}.nvmlDeviceGetHandleByIndex", return_value=None)

    vram_info = mocker.MagicMock()
    vram_info.used = 3 * 1024**3
    vram_info.total = 6 * 1024**3

    gpu_usage = mocker.MagicMock()
    gpu_usage.memory = 5
    gpu_usage.gpu = 10

    mocker.patch(f"{prefix}.nvmlDeviceGetMemoryInfo", return_value=vram_info)
    mocker.patch(f"{prefix}.nvmlDeviceGetUtilizationRates", return_value=gpu_usage)


@pytest.fixture
def cpu_metrics():
    content = {
        METRIC_CPU_COUNT: 6,
        METRIC_CPU_USAGE_PERCENT: 30.0,
        METRIC_CPU_PARALLELIZATION_PERCENT: 50.0,
        METRIC_RAM_USAGE_PERCENT: 50.0,
        METRIC_RAM_USAGE_GB: 2.0,
        METRIC_RAM_TOTAL_GB: 4.0,
        f"{METRIC_DISK_USAGE_PERCENT}/main": 50.0,
        f"{METRIC_DISK_USAGE_GB}/main": 16.0,
        f"{METRIC_DISK_TOTAL_GB}/main": 32.0,
    }
    result = {}
    for name, value in content.items():
        dpath.new(result, name, value)
    return result


def _timeserie_schema(name, value):
    return [{name: str(value), "timestamp": str, "step": "0"}]


@pytest.fixture
def cpu_timeseries():
    return {
        f"{METRIC_CPU_USAGE_PERCENT}.tsv": _timeserie_schema(
            METRIC_CPU_USAGE_PERCENT.split("/")[-1], 30.0
        ),
        f"{METRIC_CPU_PARALLELIZATION_PERCENT}.tsv": _timeserie_schema(
            METRIC_CPU_PARALLELIZATION_PERCENT.split("/")[-1], 50.0
        ),
        f"{METRIC_RAM_USAGE_PERCENT}.tsv": _timeserie_schema(
            METRIC_RAM_USAGE_PERCENT.split("/")[-1], 50.0
        ),
        f"{METRIC_RAM_USAGE_GB}.tsv": _timeserie_schema(
            METRIC_RAM_USAGE_GB.split("/")[-1], 2.0
        ),
        f"{METRIC_DISK_USAGE_PERCENT}/main.tsv": _timeserie_schema("main", 50.0),
        f"{METRIC_DISK_USAGE_GB}/main.tsv": _timeserie_schema("main", 16.0),
    }


@pytest.fixture
def gpu_timeseries():
    return {
        f"{METRIC_GPU_USAGE_PERCENT}/0.tsv": _timeserie_schema("0", 50.0),
        f"{METRIC_GPU_USAGE_PERCENT}/1.tsv": _timeserie_schema("1", 50.0),
        f"{METRIC_VRAM_USAGE_PERCENT}/0.tsv": _timeserie_schema("0", 50.0),
        f"{METRIC_VRAM_USAGE_PERCENT}/1.tsv": _timeserie_schema("1", 50.0),
        f"{METRIC_VRAM_USAGE_GB}/0.tsv": _timeserie_schema("0", 3.0),
        f"{METRIC_VRAM_USAGE_GB}/1.tsv": _timeserie_schema("1", 3.0),
    }


def test_monitor_system_is_false(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    mock_pynvml(mocker, num_gpus=0)
    system_monitor_mock = mocker.patch(
        "dvclive.live._SystemMonitor", spec=_SystemMonitor
    )
    Live(tmp_dir, save_dvc_exp=False, monitor_system=False)
    system_monitor_mock.assert_not_called()


def test_monitor_system_is_true(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    mock_pynvml(mocker, num_gpus=0)
    system_monitor_mock = mocker.patch(
        "dvclive.live._SystemMonitor", spec=_SystemMonitor
    )

    Live(tmp_dir, save_dvc_exp=False, monitor_system=True)
    system_monitor_mock.assert_called_once()


def test_all_threads_close(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    mock_pynvml(mocker, num_gpus=0)

    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=True,
    ) as live:
        first_end_spy = mocker.spy(live._system_monitor, "end")
        first_end_spy.assert_not_called()

        live.monitor_system(interval=0.01)
        first_end_spy.assert_called_once()

        second_end_spy = mocker.spy(live._system_monitor, "end")

    # check the monitoring thread is stopped
    second_end_spy.assert_called_once()


def test_ignore_non_existent_directories(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk_with_oserror(mocker)
    mock_pynvml(mocker, num_gpus=0)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        non_existent_disk = "/non-existent"
        system_monitor = _SystemMonitor(
            live=live,
            interval=0.1,
            num_samples=4,
            directories_to_monitor={"main": "/", "non-existent": non_existent_disk},
        )
        metrics = system_monitor._get_metrics()
        system_monitor.end()

    assert not Path(non_existent_disk).exists()

    assert f"{METRIC_DISK_USAGE_PERCENT}/non-existent" not in metrics
    assert f"{METRIC_DISK_USAGE_GB}/non-existent" not in metrics
    assert f"{METRIC_DISK_TOTAL_GB}/non-existent" not in metrics


@pytest.mark.timeout(2)
def test_monitor_system_metrics(tmp_dir, cpu_metrics, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    mock_pynvml(mocker, num_gpus=0)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        live.monitor_system(interval=0.05, num_samples=4)
        # wait for the metrics to be logged.
        # METRIC_DISK_TOTAL_GB is the last metric to be logged.
        while len(dpath.search(live.summary, METRIC_DISK_TOTAL_GB)) == 0:
            time.sleep(0.001)
        live.next_step()

        _, latest = parse_metrics(live)

    schema = {"step": 0, **cpu_metrics}
    assert latest == S(schema)


@pytest.mark.timeout(2)
def test_monitor_system_timeseries(tmp_dir, cpu_timeseries, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    mock_pynvml(mocker, num_gpus=0)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        live.monitor_system(interval=0.05, num_samples=4)

        # wait for the metrics to be logged.
        # METRIC_DISK_TOTAL_GB is the last metric to be logged.
        while len(dpath.search(live.summary, METRIC_DISK_TOTAL_GB)) == 0:
            time.sleep(0.001)

        live.next_step()

        timeseries, _ = parse_metrics(live)

    prefix = Path(tmp_dir) / "plots/metrics"
    schema = {str(prefix / name): value for name, value in cpu_timeseries.items()}
    assert timeseries == S(schema)


@pytest.mark.timeout(2)
def test_monitor_system_metrics_with_gpu(tmp_dir, cpu_metrics, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    mock_pynvml(mocker, num_gpus=2)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        live.monitor_system(interval=0.05, num_samples=4)
        # wait for the metrics to be logged.
        # METRIC_DISK_TOTAL_GB is the last metric to be logged.
        while len(dpath.search(live.summary, METRIC_DISK_TOTAL_GB)) == 0:
            time.sleep(0.001)
        live.next_step()

        _, latest = parse_metrics(live)

    schema = {"step": 0, **cpu_metrics}
    gpu_content = {
        METRIC_GPU_COUNT: 2,
        f"{METRIC_GPU_USAGE_PERCENT}": {"0": 50.0, "1": 50.0},
        f"{METRIC_VRAM_USAGE_PERCENT}": {"0": 50.0, "1": 50.0},
        f"{METRIC_VRAM_USAGE_GB}": {"0": 3.0, "1": 3.0},
        f"{METRIC_VRAM_TOTAL_GB}": {"0": 6.0, "1": 6.0},
    }
    for name, value in gpu_content.items():
        dpath.new(schema, name, value)
    assert latest == S(schema)


@pytest.mark.timeout(2)
def test_monitor_system_timeseries_with_gpu(
    tmp_dir, cpu_timeseries, gpu_timeseries, mocker
):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    mock_pynvml(mocker, num_gpus=2)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        live.monitor_system(interval=0.05, num_samples=4)

        # wait for the metrics to be logged.
        # METRIC_DISK_TOTAL_GB is the last metric to be logged.
        while len(dpath.search(live.summary, METRIC_DISK_TOTAL_GB)) == 0:
            time.sleep(0.001)

        live.next_step()

        timeseries, _ = parse_metrics(live)

    prefix = Path(tmp_dir) / "plots/metrics"
    schema = {str(prefix / name): value for name, value in cpu_timeseries.items()}
    schema.update({str(prefix / name): value for name, value in gpu_timeseries.items()})
    assert timeseries == S(schema)
