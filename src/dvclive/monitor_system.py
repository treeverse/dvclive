import logging
import os
from statistics import mean
from threading import Event, Thread
from typing import Union

import psutil
from funcy import merge_with

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger("dvclive")
GIGABYTES_DIVIDER = 1024.0**3

MINIMUM_CPU_USAGE_TO_BE_ACTIVE = 20

METRIC_CPU_COUNT = "system/cpu/count"
METRIC_CPU_USAGE_PERCENT = "system/cpu/usage (%)"
METRIC_CPU_PARALLELIZATION_PERCENT = "system/cpu/parallelization (%)"

METRIC_RAM_USAGE_PERCENT = "system/ram/usage (%)"
METRIC_RAM_USAGE_GB = "system/ram/usage (GB)"
METRIC_RAM_TOTAL_GB = "system/ram/total (GB)"

METRIC_DISK_USAGE_PERCENT = "system/disk/usage (%)"
METRIC_DISK_USAGE_GB = "system/disk/usage (GB)"
METRIC_DISK_TOTAL_GB = "system/disk/total (GB)"

METRIC_GPU_COUNT = "system/gpu/count"
METRIC_GPU_USAGE_PERCENT = "system/gpu/usage (%)"
METRIC_VRAM_USAGE_PERCENT = "system/vram/usage (%)"
METRIC_VRAM_USAGE_GB = "system/vram/usage (GB)"
METRIC_VRAM_TOTAL_GB = "system/vram/total (GB)"


class _SystemMonitor:
    _plot_blacklist_prefix: tuple = (
        METRIC_CPU_COUNT,
        METRIC_RAM_TOTAL_GB,
        METRIC_DISK_TOTAL_GB,
        METRIC_GPU_COUNT,
        METRIC_VRAM_TOTAL_GB,
    )

    def __init__(
        self,
        live,
        interval: float,  # seconds
        num_samples: int,
        directories_to_monitor: dict[str, str],
    ):
        self._live = live
        self._interval = self._check_interval(interval, max_interval=0.1)
        self._num_samples = self._check_num_samples(
            num_samples, min_num_samples=1, max_num_samples=30
        )
        self._disks_to_monitor = self._check_directories_to_monitor(
            directories_to_monitor
        )
        self._warn_cpu_problem = True
        self._warn_gpu_problem = True
        self._warn_disk_doesnt_exist: dict[str, bool] = {}

        self._shutdown_event = Event()
        Thread(
            target=self._monitoring_loop,
        ).start()

    def _check_interval(self, interval: float, max_interval: float) -> float:
        if interval > max_interval:
            logger.warning(
                f"System monitoring `interval` should be less than {max_interval} "
                f"seconds. Setting `interval` to {max_interval} seconds."
            )
            return max_interval
        return interval

    def _check_num_samples(
        self, num_samples: int, min_num_samples: int, max_num_samples: int
    ) -> int:
        min_num_samples = 1
        max_num_samples = 30
        if not min_num_samples < num_samples < max_num_samples:
            num_samples = max(min(num_samples, max_num_samples), min_num_samples)
            logger.warning(
                f"System monitoring `num_samples` should be between {min_num_samples} "
                f"and {max_num_samples}. Setting `num_samples` to {num_samples}."
            )
        return num_samples

    def _check_directories_to_monitor(
        self, directories_to_monitor: dict[str, str]
    ) -> dict[str, str]:
        disks_to_monitor = {}
        for disk_name, disk_path in directories_to_monitor.items():
            if disk_name != os.path.normpath(disk_name):
                raise ValueError(  # noqa: TRY003
                    "Keys for `directories_to_monitor` should be a valid name"
                    f", but got '{disk_name}'."
                )
            disks_to_monitor[disk_name] = disk_path
        return disks_to_monitor

    def _monitoring_loop(self):
        while not self._shutdown_event.is_set():
            self._metrics = {}
            last_metrics = {}
            for _ in range(self._num_samples):
                try:
                    last_metrics = self._get_metrics()
                except psutil.Error:
                    if self._warn_cpu_problem:
                        logger.exception("Failed to monitor CPU metrics")
                        self._warn_cpu_problem = False
                except NVMLError:
                    if self._warn_gpu_problem:
                        logger.exception("Failed to monitor GPU metrics")
                        self._warn_gpu_problem = False

                self._metrics = merge_with(sum, self._metrics, last_metrics)
                self._shutdown_event.wait(self._interval)
                if self._shutdown_event.is_set():
                    break
            for name, values in self._metrics.items():
                blacklisted = any(
                    name.startswith(prefix) for prefix in self._plot_blacklist_prefix
                )
                self._live.log_metric(
                    name,
                    values / self._num_samples,
                    timestamp=True,
                    plot=None if blacklisted else True,
                )

    def _get_metrics(self) -> dict[str, Union[float, int]]:
        return {
            **self._get_gpu_info(),
            **self._get_cpu_info(),
            **self._get_ram_info(),
            **self._get_disk_info(),
        }

    def _get_ram_info(self) -> dict[str, Union[float, int]]:
        ram_info = psutil.virtual_memory()
        return {
            METRIC_RAM_USAGE_PERCENT: ram_info.percent,
            METRIC_RAM_USAGE_GB: ram_info.used / GIGABYTES_DIVIDER,
            METRIC_RAM_TOTAL_GB: ram_info.total / GIGABYTES_DIVIDER,
        }

    def _get_cpu_info(self) -> dict[str, Union[float, int]]:
        num_cpus = psutil.cpu_count()
        cpus_percent = psutil.cpu_percent(percpu=True)
        return {
            METRIC_CPU_COUNT: num_cpus,
            METRIC_CPU_USAGE_PERCENT: mean(cpus_percent),
            METRIC_CPU_PARALLELIZATION_PERCENT: len(
                [
                    percent
                    for percent in cpus_percent
                    if percent >= MINIMUM_CPU_USAGE_TO_BE_ACTIVE
                ]
            )
            * 100
            / num_cpus,
        }

    def _get_disk_info(self) -> dict[str, Union[float, int]]:
        result = {}
        for disk_name, disk_path in self._disks_to_monitor.items():
            try:
                disk_info = psutil.disk_usage(disk_path)
            except OSError:
                if self._warn_disk_doesnt_exist.get(disk_name, True):
                    logger.warning(
                        f"Couldn't find directory '{disk_path}', ignoring it."
                    )
                    self._warn_disk_doesnt_exist[disk_name] = False
                continue
            disk_metrics = {
                f"{METRIC_DISK_USAGE_PERCENT}/{disk_name}": disk_info.percent,
                f"{METRIC_DISK_USAGE_GB}/{disk_name}": disk_info.used
                / GIGABYTES_DIVIDER,
                f"{METRIC_DISK_TOTAL_GB}/{disk_name}": disk_info.total
                / GIGABYTES_DIVIDER,
            }
            disk_metrics = {k.rstrip("/"): v for k, v in disk_metrics.items()}
            result.update(disk_metrics)
        return result

    def _get_gpu_info(self) -> dict[str, Union[float, int]]:
        if not GPU_AVAILABLE:
            return {}

        nvmlInit()
        num_gpus = nvmlDeviceGetCount()
        gpu_metrics = {
            "system/gpu/count": num_gpus,
        }

        for gpu_idx in range(num_gpus):
            gpu_handle = nvmlDeviceGetHandleByIndex(gpu_idx)
            memory_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            usage_info = nvmlDeviceGetUtilizationRates(gpu_handle)

            gpu_metrics.update(
                {
                    f"{METRIC_GPU_USAGE_PERCENT}/{gpu_idx}": (
                        100 * usage_info.memory / usage_info.gpu
                        if usage_info.gpu
                        else 0
                    ),
                    f"{METRIC_VRAM_USAGE_PERCENT}/{gpu_idx}": (
                        100 * memory_info.used / memory_info.total
                    ),
                    f"{METRIC_VRAM_USAGE_GB}/{gpu_idx}": (
                        memory_info.used / GIGABYTES_DIVIDER
                    ),
                    f"{METRIC_VRAM_TOTAL_GB}/{gpu_idx}": (
                        memory_info.total / GIGABYTES_DIVIDER
                    ),
                }
            )
        nvmlShutdown()
        return gpu_metrics

    def end(self):
        self._shutdown_event.set()
