"""Resource limits for the containers a task's own test suite runs in.

Both the agent's SWE sandbox (nested inside dind) and the graded test container run the
task repo's tests, and those tests size their parallelism from the CPU count they see:
pytest-xdist's `--numprocesses=auto|logical` (psutil), `multiprocessing.Pool()` /
`ProcessPoolExecutor()` (os.cpu_count()), and native thread pools in OpenBLAS, polars/rayon,
tokio and V8. On a 96-vCPU sandbox host that meant e.g. 96 xdist workers with ~200 native
threads each - blowing through the pids limit and RAM - for a suite that needs a handful.

`--cpus`/`--memory` alone do not fix that: they cap what a container *consumes*, not what it
*sees*. os.cpu_count() and psutil read /sys/devices/system/cpu/online, which Docker does not
virtualize, so a `--cpus 4` container still spawns 96 workers and then gets OOM-killed or
hangs on the pids limit. So a limited container gets three things together:

- a bind-mounted /sys/devices/system/cpu/online listing only N CPUs (os.cpu_count(), psutil,
  and through them xdist, multiprocessing pools and V8),
- explicit thread-count env vars, which every library honours regardless of how it detects
  CPUs (the env is inherited by every later `docker exec` into the container),
- hard `--cpus`/`--memory`/`--pids-limit`/`--ulimit nofile` caps as a backstop.

N and the memory cap come from SOMA_TASK_CPUS / SOMA_TASK_MEMORY, which sandbox hosts get
computed from their own CPU/RAM and concurrency (SANDBOX-ANSIBLE roles/capacity_facts); the
defaults below only apply when those are unset. SOMA_TASK_CPUS=0 disables all of it.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

DEFAULT_TASK_CPUS = 4
DEFAULT_TASK_MEMORY = "4g"
DEFAULT_TASK_PIDS_LIMIT = 1024
DEFAULT_TASK_NOFILE = 65536

CPU_ONLINE_SYSFS_PATH = "/sys/devices/system/cpu/online"

# Libraries that size a worker/thread pool from the CPU count but take an explicit override.
THREAD_COUNT_ENV_VARS = (
    "PYTEST_XDIST_AUTO_NUM_WORKERS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)

_MEMORY_PATTERN = re.compile(r"^[1-9][0-9]*[bkmg]?$", re.IGNORECASE)


@dataclass(frozen=True)
class TaskContainerLimits:
    cpus: int
    memory: str
    pids_limit: int
    nofile: int

    def cpu_online(self) -> str:
        """Contents for the fake /sys/devices/system/cpu/online (kernel cpulist format)."""
        return "0\n" if self.cpus == 1 else f"0-{self.cpus - 1}\n"

    def docker_run_args(self, *, cpu_online_file: str) -> list[str]:
        """`docker run` flags; `cpu_online_file` must hold cpu_online() on the daemon's side."""
        args = [
            "--cpus", str(self.cpus),
            "--memory", self.memory,
            "--memory-swap", self.memory,
            "--pids-limit", str(self.pids_limit),
            "--ulimit", f"nofile={self.nofile}:{self.nofile}",
            "-v", f"{cpu_online_file}:{CPU_ONLINE_SYSFS_PATH}:ro",
        ]
        for name in THREAD_COUNT_ENV_VARS:
            args.extend(["-e", f"{name}={self.cpus}"])
        return args

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _host_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _resolve(runtime_options: Mapping[str, Any] | None, name: str, env_name: str) -> Any:
    if runtime_options and name in runtime_options:
        return runtime_options[name]
    return os.getenv(env_name)


def _non_negative_int(raw_value: Any, default: int) -> int:
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def resolve_task_container_limits(
    runtime_options: Mapping[str, Any] | None = None,
) -> TaskContainerLimits | None:
    """Limits for a task test container, or None when disabled (SOMA_TASK_CPUS=0)."""
    cpus = _non_negative_int(_resolve(runtime_options, "task_cpus", "SOMA_TASK_CPUS"), DEFAULT_TASK_CPUS)
    if cpus == 0:
        return None
    # Never advertise more CPUs than this host actually has.
    cpus = min(cpus, _host_cpu_count())

    raw_memory = _resolve(runtime_options, "task_memory", "SOMA_TASK_MEMORY")
    memory = str(raw_memory).strip() if raw_memory is not None else ""
    if not _MEMORY_PATTERN.match(memory):
        memory = DEFAULT_TASK_MEMORY

    pids_limit = _non_negative_int(
        _resolve(runtime_options, "task_pids_limit", "SOMA_TASK_PIDS_LIMIT"), DEFAULT_TASK_PIDS_LIMIT
    ) or DEFAULT_TASK_PIDS_LIMIT
    nofile = _non_negative_int(
        _resolve(runtime_options, "task_nofile", "SOMA_TASK_NOFILE"), DEFAULT_TASK_NOFILE
    ) or DEFAULT_TASK_NOFILE

    return TaskContainerLimits(cpus=cpus, memory=memory, pids_limit=pids_limit, nofile=nofile)
