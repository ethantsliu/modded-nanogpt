"""Microbenchmark of ANVIL's matrix work on one GPU, record implementation vs each ANVIL_KERNEL (about a minute).

    python anvil-kernel/bench.py                          # node: one H100, CUDA-graph replays, real shapes
    python anvil-kernel/bench.py --json kbench.json --reps 500
    python anvil-kernel/bench.py --device cpu --smoke     # laptop: tiny reps, shapes / 4, numerics only

One rank's share of each bank, as the trainer runs it (world size 8): qk_bank 6 x 256x768, vo_bank 2 x 768x768,
mlp_bank 3 x 2816x768, bf16, each bank one batched call. Cases (one KBENCH line each; base = the record):
  matrix.<bank>.<kernel>   the whitening cascade alone, from the bf16 momentum: the record's kernel sequence
                           (reference_cascade) vs the Gram-space cascade. Also the Frobenius error of each against
                           the same rule in fp64 (base_exact_err / alt_exact_err). Runs on CPU too (RefOps).
  update.<bank>.<kernel>   GPU only: the whole bank update the optimizer graph captures (anvil_bank_update: rails,
                           cascade, equalizer, decay/update), record vs ANVIL_KERNEL. The error is on the parameter
                           step it applies; rails_equal says whether the velocity came out bit-identical.
  ablate.<bank>            GPU only: the whole bank update vs ANVIL_ABLATE=skip_matrix (no cascade). base - alt is
                           the cascade's matrix work inside the update.
  prim.<op>.<bank>         GPU only: single kernels. gram / poly: the record's symmetric Triton kernel (base) vs a
                           plain dense cuBLAS product (alt), i.e. what the existing symmetric trick buys (speedup < 1
                           means the record's kernel wins). small: cuBLAS baddbmm (base) vs the new symmetric
                           two-operand kernel (alt). update: the split a*X + X@B (base) vs the bare GEMM (alt).
Timing: by default each side is captured in a CUDA graph (as the trainer replays it) and timed with CUDA events
around the replay, base and alt interleaved; --no-graph times eager calls instead (adds host launch time). Median
and p10 / p90 over --reps after --warmup. peak_mb is the extra allocator peak of one eager call.
max_abs_err / max_rel_err: max |alt - base| and that over max |base|; fro_rel_err: ||alt - base||_F / ||base||_F.
KBENCH_TOTAL lines: the per-step total over the three banks and what it is worth over a 1194-step run IF the bank
updates are fully exposed (they may hide under comms: only the end-to-end A/B measures that).
Nothing here is a training run.
"""
import argparse
import ast
import json
import math
import os
import pathlib
import statistics
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _k in ("ANVIL_KERNEL", "ANVIL_ABLATE", "ANVIL_BANKS"):  # the bench switches kernels itself
    os.environ.pop(_k, None)

import torch  # noqa: E402

from track_1_short.config import ANVIL_KERNELS  # noqa: E402
from track_1_short.perf import anvil_kernel as AK  # noqa: E402

WORLD = 8
RUN_STEPS = 1194  # 1122 scheduled + 52 growth + 20 extension (config.py)
# label -> the bank's `.reshape` (model/gpt.py): [matrices, rows, cols]; a rank holds matrices / WORLD of them.
BANK_SHAPES = {"qk_bank": (48, 256, 768), "vo_bank": (16, 768, 768), "mlp_bank": (24, 2816, 768)}


def anvil_constants() -> dict:
    """ANVIL_MAPS and the scalar constants, read from optim/anvil.py's source (importing it needs Triton)."""
    tree = ast.parse((ROOT / "track_1_short" / "optim" / "anvil.py").read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            names = [target.id] if isinstance(target, ast.Name) else [e.id for e in getattr(target, "elts", [])]
            if not names or not all(n.isupper() for n in names):
                continue
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            out.update({names[0]: value} if len(names) == 1 else dict(zip(names, value)))
    return out


CONSTS = anvil_constants()
MAPS, MARGIN, EPS = CONSTS["ANVIL_MAPS"], CONSTS["FROBENIUS_MARGIN"], CONSTS["FROBENIUS_EPS"]
SLOW_BETA, SPLIT_MIN_ROWS = CONSTS["RAIL_SLOW_BETA"], CONSTS["SPLIT_BADDBMM_MIN_ROWS"]


def chunk_shape(label: str, scale: int = 1) -> tuple[int, int, int]:
    n, rows, cols = BANK_SHAPES[label]
    return n // WORLD, rows // scale, cols // scale


def synthetic(shape, spectrum: str, device, seed: int) -> torch.Tensor:
    """A batch of bf16 matrices with a set singular spectrum (gauss: i.i.d. entries), gradient-sized entries."""
    g = torch.Generator().manual_seed(seed)
    batch, rows, cols = shape
    r = min(rows, cols)
    if spectrum == "gauss":
        return (0.01 * torch.randn(shape, generator=g)).to(device).bfloat16()
    i = torch.arange(1, r + 1, dtype=torch.float32)
    s = {"p05": i ** -0.5, "p1": i ** -1.0, "lowrank": torch.where(i <= r // 8, 1.0, 1e-3)}[spectrum]
    U = torch.linalg.qr(torch.randn(batch, rows, r, generator=g).to(device)).Q
    V = torch.linalg.qr(torch.randn(batch, cols, r, generator=g).to(device)).Q
    return (0.01 * (U * s.to(device)) @ V.mT).bfloat16()


def exact_cascade(X: torch.Tensor) -> torch.Tensor:
    """The update rule in fp64: Frobenius normalization, then the six maps."""
    X = X.double()
    X = X / (torch.linalg.matrix_norm(X, keepdim=True) * MARGIN + EPS)
    tall = X.size(-2) > X.size(-1)
    for a, b, c in MAPS:
        A = X.mT @ X if tall else X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + (X @ B if tall else B @ X)
    return X


def errors(alt: torch.Tensor, base: torch.Tensor) -> dict:
    alt, base = alt.double(), base.double()
    diff = (alt - base).abs().max().item()
    return dict(max_abs_err=diff, max_rel_err=diff / max(base.abs().max().item(), 1e-300),
                fro_rel_err=((alt - base).norm() / base.norm().clamp_min(1e-300)).item())


def fro_rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x.double() - ref).norm() / ref.norm()).item()


# ----------------------------------------------------------------------------------------------------------- timing

class Timer:
    def __init__(self, device: torch.device, reps: int, warmup: int, graph: bool):
        self.device, self.reps, self.warmup = device, reps, warmup
        self.cuda = device.type == "cuda"
        self.graph = graph and self.cuda

    def _runner(self, fn):
        if not self.graph:
            return fn
        for _ in range(3):  # Triton / inductor compiles and cuBLAS workspaces happen outside the capture
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g.replay

    def peak_mb(self, fn) -> float:
        if not self.cuda:
            return float("nan")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        del out
        return peak / 2 ** 20

    def pair(self, base, alt) -> tuple[dict, dict]:
        """Interleaved timings of two callables: (median, p10, p90) in ms each."""
        runs = [self._runner(base), self._runner(alt)]
        times = [[], []]
        for rep in range(self.warmup + self.reps):
            for i in ((0, 1) if rep % 2 == 0 else (1, 0)):
                if self.cuda:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start.record()
                    runs[i]()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                else:
                    t0 = time.perf_counter()
                    runs[i]()
                    ms = (time.perf_counter() - t0) * 1e3
                if rep >= self.warmup:
                    times[i].append(ms)
        return tuple(self._stats(t) for t in times)

    @staticmethod
    def _stats(t: list[float]) -> dict:
        t = sorted(t)
        pick = lambda q: t[min(len(t) - 1, max(0, round(q * (len(t) - 1))))]
        return dict(ms=statistics.median(t), p10=pick(0.1), p90=pick(0.9))


RESULTS: list[dict] = []
FAILED: list[str] = []


def emit(name: str, shape, base: dict, alt: dict, err: dict, peak_mb: float, **extra):
    row = dict(name=name, shape="x".join(map(str, shape)), base_ms=base["ms"], alt_ms=alt["ms"],
               speedup=base["ms"] / alt["ms"] if alt["ms"] > 0 else float("nan"),
               max_rel_err=err.get("max_rel_err", float("nan")), peak_mb=peak_mb,
               base_p10=base["p10"], base_p90=base["p90"], alt_p10=alt["p10"], alt_p90=alt["p90"],
               max_abs_err=err.get("max_abs_err", float("nan")), fro_rel_err=err.get("fro_rel_err", float("nan")), **extra)
    RESULTS.append(row)
    fmt = lambda v: v if isinstance(v, str) else (f"{v:.4g}" if isinstance(v, float) else str(v))
    print("KBENCH " + " ".join(f"{k}={fmt(v)}" for k, v in row.items()), flush=True)


def guarded(name: str, fn):
    """One failing case (a kernel that does not compile, say) must not lose the others."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        print(f"KBENCH_ERROR name={name} err={type(e).__name__}:{str(e).splitlines()[0] if str(e) else ''}", flush=True)
        traceback.print_exc()


# ------------------------------------------------------------------------------------------------------------ cases

def case_matrix(timer: Timer, label: str, kernel: str, X: torch.Tensor, exact: torch.Tensor):
    groups, sym = ANVIL_KERNELS[kernel]
    split = X.size(-2) > SPLIT_MIN_ROWS
    base = lambda: AK.reference_cascade(X, AK.ops_for(X), MAPS, MARGIN, EPS, split)
    alt = lambda: AK.gram_cascade(X, AK.ops_for(X, sym), MAPS, groups, MARGIN, EPS)
    out_base, out_alt = base(), alt()
    err = errors(out_alt, out_base)
    extra = dict(base_exact_err=fro_rel(out_base, exact), alt_exact_err=fro_rel(out_alt, exact))
    del out_base, out_alt
    peak = timer.peak_mb(alt)
    tb, ta = timer.pair(base, alt)
    emit(f"matrix.{label}.{kernel}", X.shape, tb, ta, err, peak, **extra)


def make_bank(label: str, grad: torch.Tensor):
    """An AnvilBank on random state, with mid-run scalars (training.py's defaults, past the rail engage step)."""
    from track_1_short.optim.anvil import AnvilBank
    dev, shape = grad.device, tuple(grad.shape)
    g = torch.Generator().manual_seed(7)
    rows, cols = shape[-2:]
    lane = (*shape[:-1], 1) if rows >= cols else (*shape[:-2], 1, cols)
    scalar = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)
    return AnvilBank(
        label=label, grad=grad, velocity=torch.zeros((2, *shape), dtype=torch.float32, device=dev),
        lane_energy=torch.zeros(lane, dtype=torch.float32, device=dev),
        mantissa=torch.zeros(shape, dtype=torch.uint16, device=dev),
        p_slice=(0.02 * torch.randn(shape, generator=g)).to(dev).bfloat16(),
        momentum=scalar(0.93), eff_wd=scalar(2.25 * 0.023), fast_beta=scalar(CONSTS["RAIL_FAST_BETA"]),
        fast_weight=scalar(CONSTS["RAIL_FAST_WEIGHT"]),
        eff_lr=torch.full((shape[0], 1, 1), 0.023 * max(1.0, rows / cols) ** 0.5, dtype=torch.float32, device=dev),
        beta2=0.9, split_baddbmm=rows > SPLIT_MIN_ROWS, red_dim=-1 if rows >= cols else -2)


def master(bank) -> torch.Tensor:
    """The fp32 master weights (bf16 parameter bits over the uint16 mantissa)."""
    hi = bank.p_slice.view(torch.uint16).to(torch.int32) << 16
    return (hi | bank.mantissa.to(torch.int32)).view(torch.float32).clone()


def flagged_update(bank, kernel: str = "", ablate: str = ""):
    from track_1_short.optim.anvil import anvil_bank_update

    def run():
        AK.configure(kernel, ablate, (bank.label,))
        try:
            anvil_bank_update(bank)
        finally:
            AK.configure()
    return run


def case_update(timer: Timer, label: str, grad: torch.Tensor, kernel: str = "", ablate: str = ""):
    from track_1_short.perf.cuda_graphs.capture_support import bits_equal, restore, snapshot
    bank = make_bank(label, grad)
    base, alt = flagged_update(bank), flagged_update(bank, kernel, ablate)
    # Two warm steps so the rails hold state, then one step of each from the same state.
    for _ in range(2):
        base()
    start, w0 = snapshot(bank.mutated()), master(bank)
    base()
    step_base, vel_base = master(bank) - w0, bank.velocity.clone()
    restore(bank.mutated(), start)
    alt()
    step_alt, rails_equal = master(bank) - w0, int(bits_equal(bank.velocity, vel_base))
    restore(bank.mutated(), start)
    err = errors(step_alt, step_base) if kernel else {}
    peak = timer.peak_mb(alt)
    tb, ta = timer.pair(base, alt)
    name = f"update.{label}.{kernel}" if kernel else f"ablate.{label}"
    emit(name, grad.shape, tb, ta, err, peak, rails_equal=rails_equal)


def case_prims(timer: Timer, label: str, X: torch.Tensor):
    ops, ops_sym = AK.ops_for(X), AK.ops_for(X, True)
    tall = X.size(-2) > X.size(-1)
    a, b, c = MAPS[0]
    A = ops.gram(X, tall)
    A = (A.float() / AK._frobenius_scale(A, MARGIN, EPS).square()).bfloat16()
    B = ops.poly(A, b, c)
    Xn = (X.float() / torch.linalg.matrix_norm(X.float(), keepdim=True)).bfloat16()

    def one(name, base, alt):
        def run():
            err = errors(alt(), base())
            tb, ta = timer.pair(base, alt)
            emit(f"prim.{name}.{label}", X.shape, tb, ta, err, timer.peak_mb(alt))
        guarded(f"prim.{name}.{label}", run)

    one("gram", lambda: ops.gram(Xn, tall), lambda: torch.matmul(Xn.mT, Xn) if tall else torch.matmul(Xn, Xn.mT))
    one("poly", lambda: ops.poly(A, b, c), lambda: torch.baddbmm(A, A, A, beta=b, alpha=c))
    one("small", lambda: ops.small(A, B, a), lambda: ops_sym.small(A, B, a))
    one("update", lambda: ops.update(Xn, B, a, tall, True), lambda: ops.apply(Xn, B, tall))


def totals(kernels: list[str], labels: list[str], device: torch.device):
    by_name = {r["name"]: r for r in RESULTS}
    line = lambda **kw: print("KBENCH_TOTAL " + " ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)
    for kind in ("update", "matrix"):
        for kernel in kernels:
            rows = {lb: by_name.get(f"{kind}.{lb}.{kernel}") for lb in labels}
            if any(r is None for r in rows.values()):
                continue
            base = sum(r["base_ms"] for r in rows.values())
            for banks, pick in (("mlp_bank", lambda lb, r: r["alt_ms"] if lb == "mlp_bank" else r["base_ms"]),
                                ("all", lambda lb, r: r["alt_ms"]),
                                ("best_per_bank", lambda lb, r: min(r["alt_ms"], r["base_ms"]))):
                if banks == "mlp_bank" and "mlp_bank" not in rows:
                    continue
                alt = sum(pick(lb, r) for lb, r in rows.items())
                # A CPU time says nothing about the GPU: no per-run projection from it.
                per_run = (base - alt) * RUN_STEPS / 1e3 if device.type == "cuda" else float("nan")
                line(kind=kind, kernel=kernel, banks=banks, device=device.type, base_ms_per_step=base,
                     alt_ms_per_step=alt, saved_ms_per_step=base - alt,
                     saved_s_per_run_if_exposed=per_run, steps=RUN_STEPS)
    rows = [by_name.get(f"ablate.{lb}") for lb in labels]
    if all(r is not None for r in rows):
        base, alt = sum(r["base_ms"] for r in rows), sum(r["alt_ms"] for r in rows)
        line(kind="matrix_share", device=device.type, update_ms_per_step=base, skip_matrix_ms_per_step=alt,
             matrix_ms_per_step=base - alt, matrix_s_per_run_if_exposed=(base - alt) * RUN_STEPS / 1e3,
             steps=RUN_STEPS, per_bank=",".join(f"{lb}:{r['base_ms'] - r['alt_ms']:.4g}" for lb, r in zip(labels, rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="2 reps, 1 warmup, shapes / 4 (unless --scale is given)")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--scale", type=int, help="divide every matrix dimension by this (1 = the real shapes)")
    ap.add_argument("--kernels", default=",".join(ANVIL_KERNELS))
    ap.add_argument("--banks", default=",".join(BANK_SHAPES))
    ap.add_argument("--shapes", help="size sweep instead of the record's banks: comma list of NxROWSxCOLS (N matrices per rank), "
                                     "e.g. 3x2816x768,1x16384x4096; each becomes a bank labelled sweep_NxROWSxCOLS")
    ap.add_argument("--spectrum", default="p05", choices=("gauss", "p05", "p1", "lowrank"))
    ap.add_argument("--no-graph", action="store_true", help="time eager calls instead of CUDA-graph replays")
    ap.add_argument("--json", help="also write every row to this file")
    args = ap.parse_args()
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    reps = args.reps or (2 if args.smoke else 300 if cuda else 3)
    warmup = args.warmup if args.warmup is not None else (1 if args.smoke else 30 if cuda else 1)
    scale = args.scale or (4 if args.smoke else 1)
    kernels = [k for k in args.kernels.split(",") if k]
    labels = [b for b in args.banks.split(",") if b]
    if args.shapes:  # a size sweep: register each shape as its own bank (BANK_SHAPES holds the all-rank count)
        labels = []
        for spec in args.shapes.split(","):
            n, rows, cols = (int(v) for v in spec.lower().split("x"))
            BANK_SHAPES[f"sweep_{n}x{rows}x{cols}"] = (n * WORLD, rows, cols)
            labels.append(f"sweep_{n}x{rows}x{cols}")
    assert all(k in ANVIL_KERNELS for k in kernels) and all(b in BANK_SHAPES for b in labels)
    timer = Timer(device, reps, warmup, graph=not args.no_graph)
    if not cuda:
        torch.set_num_threads(min(torch.get_num_threads(), int(os.environ.get("OMP_NUM_THREADS", "2"))))
    print(f"KBENCH_INFO device={device.type}" + (f" gpu={torch.cuda.get_device_name(device).replace(' ', '_')}" if cuda else "")
          + f" torch={torch.__version__} reps={reps} warmup={warmup} scale={scale} spectrum={args.spectrum} "
          f"mode={'graph' if timer.graph else 'eager'} maps={len(MAPS)} world={WORLD}", flush=True)
    if not cuda:
        print("KBENCH_INFO note=cpu_times_are_a_smoke_test_only(RefOps_fp32_emulation);update/ablate/prim_cases_need_cuda", flush=True)

    with torch.no_grad():
        for seed, label in enumerate(labels):
            X = synthetic(chunk_shape(label, scale), args.spectrum, device, seed)
            exact = exact_cascade(X)
            for kernel in kernels:
                guarded(f"matrix.{label}.{kernel}", lambda: case_matrix(timer, label, kernel, X, exact))
            if cuda:
                for kernel in kernels:
                    guarded(f"update.{label}.{kernel}", lambda: case_update(timer, label, X, kernel=kernel))
                guarded(f"ablate.{label}", lambda: case_update(timer, label, X, ablate="skip_matrix"))
                case_prims(timer, label, X)
            del X, exact
    totals(kernels, labels, device)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(args=vars(args), rows=RESULTS, failed=FAILED), f, indent=1)
    print(f"KBENCH_DONE cases={len(RESULTS)} failed={len(FAILED)}" + (f" failed_names={','.join(FAILED)}" if FAILED else ""),
          flush=True)
    return 1 if FAILED or not all(math.isfinite(r["alt_ms"]) for r in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
