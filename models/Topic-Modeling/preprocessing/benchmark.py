import time
import tracemalloc
import psutil
import os
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class BenchmarkResult:
    label: str
    wall_time_s: float
    cpu_user_s: float
    cpu_system_s: float
    peak_rss_mb: float        # OS-level peak resident memory (psutil)
    tracemalloc_peak_mb: float  # Python heap peak (tracemalloc)
    delta_rss_mb: float       # RSS growth over the block

    def report(self):
        print(f"\n── {self.label} ──")
        print(f"  Wall time        : {self.wall_time_s:.3f}s")
        print(f"  CPU user/sys     : {self.cpu_user_s:.3f}s / {self.cpu_system_s:.3f}s")
        print(f"  Peak RSS         : {self.peak_rss_mb:.1f} MB")
        print(f"  RSS delta        : {self.delta_rss_mb:+.1f} MB")
        print(f"  Tracemalloc peak : {self.tracemalloc_peak_mb:.1f} MB")


@contextmanager
def benchmark(label: str):
    proc = psutil.Process(os.getpid())

    # Baselines
    rss_before = proc.memory_info().rss / 1024 ** 2
    cpu_before = proc.cpu_times()
    tracemalloc.start()
    wall_start = time.perf_counter()

    result = BenchmarkResult(label, 0, 0, 0, 0, 0, 0)
    try:
        yield result
    finally:
        wall_end = time.perf_counter()
        cpu_after = proc.cpu_times()
        _, tracemalloc_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = proc.memory_info().rss / 1024 ** 2

        result.wall_time_s = wall_end - wall_start
        result.cpu_user_s = cpu_after.user - cpu_before.user
        result.cpu_system_s = cpu_after.system - cpu_before.system
        result.peak_rss_mb = rss_after  # conservative — see note below
        result.delta_rss_mb = rss_after - rss_before
        result.tracemalloc_peak_mb = tracemalloc_peak / 1024 ** 2
        result.report()

# import threading

# @contextmanager
# def benchmark(label: str):
#     proc = psutil.Process(os.getpid())
#     rss_before = proc.memory_info().rss / 1024 ** 2

#     peak_rss = [rss_before]
#     stop_event = threading.Event()

#     def _poll():
#         while not stop_event.is_set():
#             peak_rss[0] = max(peak_rss[0], proc.memory_info().rss / 1024 ** 2)
#             stop_event.wait(timeout=0.05)   # poll every 50ms

#     poller = threading.Thread(target=_poll, daemon=True)
#     poller.start()
#     # ... rest of context manager
#     stop_event.set()
#     poller.join()
#     result.peak_rss_mb = peak_rss[0]