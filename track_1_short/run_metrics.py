"""`METRIC ...` log lines of RUN_METRICS=1 / VAL_CURVE=N runs (config.py).

Pure Python (no torch): the trainer passes plain numbers in, after the timed loop or with the clock stopped.
Every line is `METRIC <name> key=value ...`, on the console and in the run log:

    METRIC run_config timing=timing_run|curve_run RUN_METRICS=1 VAL_CURVE=0 seed=3 world_size=8 total_steps=1194
    METRIC val step=250 total=1194 train_time_s=8.4321 val_loss=4.123456 final=0
    METRIC step_latency scope=all steps=1-1193 n=1193 mean_ms=33.512 median_ms=33.100 p95_ms=41.250 max_ms=48.000 clock=host_nosync
    METRIC step_latency scope=stage0 steps=1-319 n=319 ...
    METRIC peak_mem rank=0 allocated_mib=61234 reserved_mib=70000 warmup_allocated_mib=63000 warmup_reserved_mib=70000
    METRIC peak_mem rank=max ...

step_latency: the difference of consecutive end-of-step readings of the training clock, taken on the host
without a synchronize (the host queues CUDA-graph replays ahead of the device, so one step's value is the
host's pacing; sums and medians over many steps follow the device). Step 0 (no reading before it; the
prefix-table build is in it) and any step that began with a validation are left out.
peak_mem: torch.cuda.max_memory_allocated / max_memory_reserved, per rank; the plain keys cover the timed
loop (counters reset at clock start), the warmup_ keys everything before it (compile, warmup, captures).
"""
from collections.abc import Sequence

MIB = 1024 * 1024


def percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (q in [0, 1]) of an ascending list."""
    if not sorted_xs:
        return float("nan")
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def step_latencies(clock_ms: Sequence[float], validated_steps=()) -> list[tuple[int, float]]:
    """(step, ms) from clock_ms[s] = training time at the end of step s; see the module docstring for the gaps."""
    skip = set(validated_steps)
    return [(s, clock_ms[s] - clock_ms[s - 1]) for s in range(1, len(clock_ms)) if s not in skip]


def _latency_line(scope: str, items: list[tuple[int, float]]) -> str:
    xs = sorted(ms for _, ms in items)
    return (f"METRIC step_latency scope={scope} steps={items[0][0]}-{items[-1][0]} n={len(xs)} "
            f"mean_ms={sum(xs) / len(xs):.3f} median_ms={percentile(xs, 0.5):.3f} p95_ms={percentile(xs, 0.95):.3f} "
            f"max_ms={xs[-1]:.3f} clock=host_nosync")


def step_latency_lines(clock_ms: Sequence[float], validated_steps=(), boundaries=()) -> list[str]:
    """One line over all steps, then one per schedule stage (boundaries: [(start, end), ...])."""
    items = step_latencies(clock_ms, validated_steps)
    if not items:
        return []
    lines = [_latency_line("all", items)]
    for i, (start, end) in enumerate(boundaries):
        stage = [(s, ms) for s, ms in items if start <= s < end]
        if stage:
            lines.append(_latency_line(f"stage{i}", stage))
    return lines


def clock_split_line(clock_ms: Sequence[float], total_ms: float) -> str:
    """Where the clock goes outside the per-step latencies: step 0 (with the prefix-table build before it) and the
    tail after the last training step's host reading (the device backlog, the final flush and the tail-average
    ship, up to the synchronize that stops the clock). clock_ms[s] = training time at the end of step s."""
    return (f"METRIC clock_split step0_ms={clock_ms[0]:.3f} loop_end_ms={clock_ms[-1]:.3f} "
            f"tail_ms={total_ms - clock_ms[-1]:.3f} total_ms={total_ms:.3f} clock=host_nosync")


def run_config_line(run_metrics: bool, val_curve: int, seed, world_size: int, total_steps: int) -> str:
    return (f"METRIC run_config timing={'curve_run' if val_curve else 'timing_run'} RUN_METRICS={int(run_metrics)} "
            f"VAL_CURVE={val_curve} seed={'random' if seed is None else seed} world_size={world_size} total_steps={total_steps}")


def val_line(step: int, total_steps: int, train_time_ms: float, val_loss: float) -> str:
    return (f"METRIC val step={step} total={total_steps} train_time_s={train_time_ms / 1000:.4f} "
            f"val_loss={val_loss:.6f} final={int(step == total_steps)}")


def peak_mem_lines(rows: Sequence[Sequence[int]]) -> list[str]:
    """rows[rank] = (allocated, reserved, warmup_allocated, warmup_reserved) in bytes -> a line per rank and the max."""
    keys = ("allocated_mib", "reserved_mib", "warmup_allocated_mib", "warmup_reserved_mib")

    def line(rank, row):
        return f"METRIC peak_mem rank={rank} " + " ".join(f"{k}={int(v) // MIB}" for k, v in zip(keys, row))

    return [line(r, row) for r, row in enumerate(rows)] + [line("max", [max(col) for col in zip(*rows)])]
