import time

import torch


class StepPhaseProfiler:
    """Step 级 phase 计时器，混合使用 CPU wall time 和 CUDA events。"""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._cpu_starts = {}
        self._cpu_durations = {}
        self._gpu_events = {}
        self._step_start_time = None

    def start_step(self):
        if not self.enabled:
            return
        self._cpu_starts.clear()
        self._cpu_durations.clear()
        self._gpu_events.clear()
        self._step_start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def add_cpu_phase(self, name: str, elapsed_ms: float):
        if not self.enabled:
            return
        self._cpu_durations[name] = self._cpu_durations.get(name, 0.0) + elapsed_ms

    def start_cpu_phase(self, name: str):
        if not self.enabled:
            return None
        start_time = time.perf_counter()
        self._cpu_starts[name] = start_time
        return start_time

    def end_cpu_phase(self, name: str, start_time=None):
        if not self.enabled:
            return
        if start_time is None:
            start_time = self._cpu_starts.pop(name, None)
        else:
            self._cpu_starts.pop(name, None)
        if start_time is None:
            return
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._cpu_durations[name] = self._cpu_durations.get(name, 0.0) + elapsed_ms

    def start_gpu_phase(self, name: str):
        if not self.enabled:
            return None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._gpu_events[name] = (start_event, end_event)
        return name

    def end_gpu_phase(self, name: str):
        if not self.enabled:
            return
        events = self._gpu_events.get(name)
        if events is None:
            return
        _, end_event = events
        end_event.record()

    def finalize_step(self):
        if not self.enabled:
            return {}, {}

        torch.cuda.synchronize()

        phase_times_ms = dict(self._cpu_durations)
        gpu_total_ms = 0.0
        for name, (start_event, end_event) in self._gpu_events.items():
            elapsed_ms = start_event.elapsed_time(end_event)
            phase_times_ms[name] = phase_times_ms.get(name, 0.0) + elapsed_ms
            gpu_total_ms += elapsed_ms

        core_wall_ms = 0.0
        if self._step_start_time is not None:
            core_wall_ms = (time.perf_counter() - self._step_start_time) * 1000.0
        phase_times_ms["step_core_wall"] = core_wall_ms
        phase_times_ms["idle_sync"] = max(0.0, core_wall_ms - gpu_total_ms)

        memory_stats_mb = {
            "alloc_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "peak_alloc_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }
        return phase_times_ms, memory_stats_mb


class PhaseLogAccumulator:
    """按日志窗口累计 phase 计时和显存统计。"""

    def __init__(self):
        self.phase_totals = {}
        self.memory_max = {}
        self.steps = 0

    def update(self, phase_times_ms, memory_stats_mb):
        self.steps += 1
        for name, value in phase_times_ms.items():
            self.phase_totals[name] = self.phase_totals.get(name, 0.0) + value
        for name, value in memory_stats_mb.items():
            self.memory_max[name] = max(self.memory_max.get(name, 0.0), value)

    def average_times(self):
        if self.steps == 0:
            return {}
        return {
            name: total / self.steps
            for name, total in self.phase_totals.items()
        }

    def reset(self):
        self.phase_totals.clear()
        self.memory_max.clear()
        self.steps = 0
