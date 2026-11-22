"""Tests for ANVIL_KERNEL / ANVIL_ABLATE (track_1_short/perf/anvil_kernel.py, config.py) and bench.py.

    python -m pytest anvil-kernel/test_optim_kernel.py -q        # laptop (CPU, no Triton) and the node (+ GPU)

CPU: each kernel against the record's evaluation order and against the fp64 rule, on the banks' aspect ratios; the
flags default off and the record's cascade is untouched; flag parsing; the ablation; the experiment file.
With CUDA (the node): the Triton symmetric two-operand kernel against baddbmm, the Gram-space cascade on the real
kernels against the CPU reference, and a bank update captured in a CUDA graph under each flag.
"""
import ast
import hashlib
import importlib
import os
import pathlib
import subprocess
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
for _k in ("ANVIL_KERNEL", "ANVIL_ABLATE", "ANVIL_BANKS"):
    os.environ.pop(_k, None)

import bench  # noqa: E402
from track_1_short import config  # noqa: E402
from track_1_short.perf import anvil_kernel as AK  # noqa: E402

MAPS, MARGIN, EPS = bench.MAPS, bench.MARGIN, bench.EPS
KERNELS = sorted(config.ANVIL_KERNELS)
# The banks' per-rank shapes / 8: qk (wide, 1:3), vo (square), mlp (tall, 11:3).
SHAPES = {"qk_bank": (6, 32, 96), "vo_bank": (2, 96, 96), "mlp_bank": (3, 352, 96)}
HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs a GPU and Triton (runs on the node)")


@pytest.fixture(autouse=True)
def _flags_off():
    AK.configure()
    yield
    AK.configure()


def _src(*parts) -> str:
    return ROOT.joinpath(*parts).read_text()


def _function_source(source: str, name: str) -> str:
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


# ------------------------------------------------------------------------------------------------------- numerics

@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("label", list(SHAPES))
@pytest.mark.parametrize("spectrum", ["gauss", "p05", "p1"])
def test_kernel_matches_record_cascade(kernel, label, spectrum):
    """Same update rule: the alternative is as close to the record's bf16 result as two bf16 evaluations of one
    rule can be, and no further from the fp64 rule than a small multiple of the record's own distance."""
    groups, sym = config.ANVIL_KERNELS[kernel]
    X = bench.synthetic(SHAPES[label], spectrum, torch.device("cpu"), seed=3)
    exact = bench.exact_cascade(X)
    base = AK.reference_cascade(X, AK.RefOps(), MAPS, MARGIN, EPS)
    alt = AK.gram_cascade(X, AK.RefOps(sym), MAPS, groups, MARGIN, EPS)
    assert alt.dtype == torch.bfloat16 and alt.shape == X.shape and torch.isfinite(alt.float()).all()
    base_exact, alt_exact = bench.fro_rel(base, exact), bench.fro_rel(alt, exact)
    err = bench.errors(alt, base)
    # Bounds from the measured table in README.md (worst cases there: 0.9% / 2x for gram222*, 1.4% / 2.5x for
    # gram33*; the heavy-tailed p1 spectrum is noisier for the record too), with margin.
    assert alt_exact < 3.0 * base_exact + 2e-3, (alt_exact, base_exact)
    assert err["fro_rel_err"] < 2.0 * base_exact + 0.02, err
    # The singular values land where the record's do: every one within the maps' own ripple of 1.
    s = torch.linalg.svdvals(alt.double())
    s_base = torch.linalg.svdvals(base.double())
    assert (s.max() - s_base.max()).abs() < 0.05


def test_single_map_groups_reduce_to_the_record_formula():
    """groups of one map: X <- X @ (a*I + B) per map, i.e. the record's a*X + X@B with only the normalization folded."""
    X = bench.synthetic(SHAPES["mlp_bank"], "p05", torch.device("cpu"), seed=5)
    base = AK.reference_cascade(X, AK.RefOps(), MAPS, MARGIN, EPS)
    alt = AK.gram_cascade(X, AK.RefOps(), MAPS, (1,) * len(MAPS), MARGIN, EPS)
    exact = bench.exact_cascade(X)
    assert bench.fro_rel(alt, exact) < 2.0 * bench.fro_rel(base, exact) + 1e-3


def test_chaining_all_six_maps_is_the_documented_failure():
    """Why the kernels are grouped: one group of six amplifies the bf16 Gram's noise on a heavy-tailed spectrum."""
    X = bench.synthetic(SHAPES["mlp_bank"], "p1", torch.device("cpu"), seed=1)
    exact = bench.exact_cascade(X)
    grouped = AK.gram_cascade(X, AK.RefOps(), MAPS, (2, 2, 2), MARGIN, EPS)
    chained = AK.gram_cascade(X, AK.RefOps(), MAPS, (6,), MARGIN, EPS)
    assert bench.fro_rel(grouped, exact) < 0.1
    assert not bench.fro_rel(chained, exact) < 1.0  # diverges (or NaN)


def test_wide_is_the_transpose_of_tall():
    X = bench.synthetic((2, 160, 48), "p05", torch.device("cpu"), seed=2)
    for fn in (lambda x: AK.gram_cascade(x, AK.RefOps(), MAPS, (2, 2, 2), MARGIN, EPS),
               lambda x: AK.reference_cascade(x, AK.RefOps(), MAPS, MARGIN, EPS)):
        tall, wide = fn(X), fn(X.mT.contiguous()).mT
        assert bench.errors(wide, tall)["fro_rel_err"] < 0.02


def test_gram_cascade_leaves_its_input_alone_and_handles_2d():
    X = bench.synthetic((1, 96, 32), "gauss", torch.device("cpu"), seed=4)
    keep = X.clone()
    out3 = AK.gram_cascade(X, AK.RefOps(), MAPS, (2, 2, 2), MARGIN, EPS)
    assert torch.equal(X, keep)
    out2 = AK.gram_cascade(X[0], AK.RefOps(), MAPS, (2, 2, 2), MARGIN, EPS)
    assert bench.errors(out2, out3[0])["fro_rel_err"] < 1e-2


# ---------------------------------------------------------------------------------- rails, ablation, bank dispatch

class _Bank:
    """The AnvilBank fields bank_cascade reads (optim/anvil.py cannot be imported without Triton)."""

    def __init__(self, label="mlp_bank", shape=(3, 88, 24), seed=0):
        g = torch.Generator().manual_seed(seed)
        self.label = label
        self.grad = (0.01 * torch.randn(shape, generator=g)).bfloat16()
        self.velocity = 0.01 * torch.randn((2, *shape), generator=g)
        self.momentum, self.fast_beta, self.fast_weight = torch.tensor(0.93), torch.tensor(0.85), torch.tensor(0.4385)


def _rails_expected(bank):
    g = bank.grad.float()
    v0 = bank.velocity[0] + (1 - 0.85) * (g - bank.velocity[0])
    v1 = bank.velocity[1] + (1 - bench.SLOW_BETA) * (g - bank.velocity[1])
    blend = 0.4385 * v0 + (1 - 0.4385) * v1
    return v0, v1, g + 0.93 * (blend - g)


def test_rails_body_is_anvil_cascades_prepass():
    """Textually: the pre-pass lines of anvil_cascade, with RAIL_SLOW_BETA as an argument. Numerically: the formula."""
    upstream = _function_source(_src("track_1_short", "optim", "anvil.py"), "anvil_cascade")
    ours = _function_source(_src("track_1_short", "perf", "anvil_kernel.py"), "_rails_body")
    for line in ("grad_chunk = grad_chunk.float()", "momentum = momentum_t.to(grad_chunk.dtype)",
                 "velocity[0].lerp_(grad_chunk, 1 - fast_beta_t.to(grad_chunk.dtype))",
                 "w = fast_weight_t.to(grad_chunk.dtype)", "blend = w * velocity[0] + (1 - w) * velocity[1]",
                 "g = grad_chunk.lerp_(blend, momentum)"):
        assert line in upstream and line in ours, line
    assert "velocity[1].lerp_(grad_chunk, 1 - RAIL_SLOW_BETA)" in upstream
    assert "velocity[1].lerp_(grad_chunk, 1 - slow_beta)" in ours
    bank = _Bank()
    grad_before = bank.grad.clone()
    v0, v1, g = _rails_expected(bank)
    X = AK.rails(bank.grad, bank.velocity, bank.momentum, bank.fast_beta, bank.fast_weight, bench.SLOW_BETA)
    assert torch.equal(bank.grad, grad_before)  # the bf16 gradient buffer is only read
    assert torch.allclose(bank.velocity[0], v0, atol=1e-7) and torch.allclose(bank.velocity[1], v1, atol=1e-7)
    assert X.dtype == torch.bfloat16 and torch.allclose(X.float(), g, rtol=2 ** -7, atol=1e-6)


def test_bank_cascade_kernel_and_ablation():
    for kernel in KERNELS:
        bank, twin = _Bank(), _Bank()
        AK.configure(kernel, banks=("mlp_bank",))
        out = AK.bank_cascade(bank, MAPS, bench.SLOW_BETA, MARGIN, EPS)
        X = AK.rails(twin.grad, twin.velocity, twin.momentum, twin.fast_beta, twin.fast_weight, bench.SLOW_BETA)
        assert torch.equal(bank.velocity, twin.velocity)
        base = AK.reference_cascade(X, AK.RefOps(), MAPS, MARGIN, EPS)
        assert bench.errors(out, base)["fro_rel_err"] < 0.05
    # The ablation: same rails, and the update is the Frobenius-normalized momentum (norm 1 / margin), nothing more.
    bank, twin = _Bank(), _Bank()
    AK.configure(ablate="skip_matrix", banks=config.ANVIL_BANK_LABELS)
    out = AK.bank_cascade(bank, MAPS, bench.SLOW_BETA, MARGIN, EPS)
    X = AK.rails(twin.grad, twin.velocity, twin.momentum, twin.fast_beta, twin.fast_weight, bench.SLOW_BETA)
    assert torch.equal(bank.velocity, twin.velocity)
    norms = torch.linalg.matrix_norm(out.float())
    assert torch.allclose(norms, torch.full_like(norms, 1 / MARGIN), rtol=5e-3)
    cos = (out.float() * X.float()).sum((-2, -1)) / (norms * torch.linalg.matrix_norm(X.float()))
    assert (cos > 0.9999).all()


# ------------------------------------------------------------------------------------- flags: off = the record

# sha256 of the record's text (upstream e33c471): the compiled cascade and its Triton kernels are not edited.
ANVIL_CASCADE_SHA = "d90995e5fd6679a1e85a617e41c60409d005945e2d6a9ef4ebdee1971724e38a"
POLAR_EXPRESS_SHA = "14ee03aa049a3047209de0d591b1e30386ed2ac0cbcdfd0f9dcf3520f7137daa"


def test_flags_default_off_and_record_path_untouched():
    assert config.ANVIL_KERNEL == "" and config.ANVIL_ABLATE == "" and config.ANVIL_BANKS == frozenset()
    assert AK.ANVIL_KERNEL == "" and AK.ANVIL_ABLATE == ""
    assert not any(AK.applies(label) for label in config.ANVIL_BANK_LABELS)
    AK.configure(banks=("mlp_bank",))  # banks without a flag select nothing (the bench's base side does this)
    assert not AK.applies("mlp_bank")
    AK.configure("gram222", banks=("mlp_bank",))
    assert AK.applies("mlp_bank") and not AK.applies("vo_bank")
    AK.configure()
    anvil = _src("track_1_short", "optim", "anvil.py")
    assert hashlib.sha256(_function_source(anvil, "anvil_cascade").encode()).hexdigest() == ANVIL_CASCADE_SHA
    assert hashlib.sha256(_src("track_1_short", "perf", "kernels", "polar_express.py").encode()).hexdigest() == POLAR_EXPRESS_SHA
    # anvil_bank_update: one host-side test, and the record's call unchanged in its else branch.
    update = next(n for n in ast.parse(anvil).body if isinstance(n, ast.FunctionDef) and n.name == "anvil_bank_update")
    branch = next(n for n in update.body if isinstance(n, ast.If))
    assert ast.unparse(branch.test) == "anvil_kernel.applies(bank.label)"
    assert ast.unparse(branch.orelse[0]) == (
        "v_chunk = anvil_cascade(bank.grad, bank.velocity, bank.momentum, split_baddbmm=bank.split_baddbmm, "
        "fast_beta_t=bank.fast_beta, fast_weight_t=bank.fast_weight)")
    assert "anvil_kernel.bank_cascade(bank, ANVIL_MAPS, RAIL_SLOW_BETA, FROBENIUS_MARGIN, FROBENIUS_EPS)" in ast.unparse(branch.body[0])
    # Off means no Triton import of the new kernel and no torch.compile call at import time.
    kernel_module = _src("track_1_short", "perf", "anvil_kernel.py")
    top_imports = [ast.unparse(n) for n in ast.parse(kernel_module).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("triton" in line or "anvil_chain" in line or "optim" in line for line in top_imports)


def _reload_config(monkeypatch, **env):
    for name in ("ANVIL_KERNEL", "ANVIL_ABLATE", "ANVIL_BANKS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return importlib.reload(config)


def test_flags_are_registered_and_validated(monkeypatch):
    try:
        c = _reload_config(monkeypatch, ANVIL_KERNEL="gram222")
        assert c.ANVIL_KERNEL == "gram222" and c.ANVIL_BANKS == {"mlp_bank"} and c.ANVIL_ABLATE == ""
        c = _reload_config(monkeypatch, ANVIL_KERNEL="GRAM33s", ANVIL_BANKS="qk_bank, mlp_bank")
        assert c.ANVIL_KERNEL == "gram33s" and c.ANVIL_BANKS == {"qk_bank", "mlp_bank"}
        c = _reload_config(monkeypatch, ANVIL_ABLATE="skip_matrix")
        assert c.ANVIL_ABLATE == "skip_matrix" and c.ANVIL_BANKS == set(c.ANVIL_BANK_LABELS)
        for off in ("", "0", "off"):
            c = _reload_config(monkeypatch, ANVIL_KERNEL=off, ANVIL_ABLATE=off)
            assert c.ANVIL_KERNEL == "" and c.ANVIL_ABLATE == "" and c.ANVIL_BANKS == frozenset()
        c = _reload_config(monkeypatch, ANVIL_BANKS="vo_bank")  # without a kernel it selects nothing
        assert c.ANVIL_BANKS == frozenset()
        for bad in (dict(ANVIL_KERNEL="gram6"), dict(ANVIL_ABLATE="1"), dict(ANVIL_KERNEL="gram222", ANVIL_BANKS="moe_bank"),
                    dict(ANVIL_KERNEL="gram222", ANVIL_ABLATE="skip_matrix")):
            with pytest.raises(AssertionError):
                _reload_config(monkeypatch, **bad)
    finally:
        _reload_config(monkeypatch)
    assert config.ANVIL_KERNEL == ""
    for groups, sym in config.ANVIL_KERNELS.values():
        assert sum(groups) == len(MAPS) and max(groups) <= 3 and isinstance(sym, bool)
    doc = config.__doc__
    assert "ANVIL_KERNEL / ANVIL_ABLATE / ANVIL_BANKS" in doc
    trainer = _src("train_gpt.py")
    assert "INVALID_FOR_RECORD(anvil_ablate)" in trainer and "anvil_kernel.describe()" in trainer
    AK.configure(ablate="skip_matrix", banks=config.ANVIL_BANK_LABELS)
    assert "NOT A VALID TRAINING RUN" in AK.describe()


def test_bench_geometry_matches_the_model():
    gpt = _src("track_1_short", "model", "gpt.py")
    assert "MLP_HIDDEN_DIM = 2816" in gpt and "NUM_MLP_SLOTS = 12" in gpt
    assert "(2 * NUM_MLP_SLOTS, self.mlp_hdim, model_dim)" in gpt and "(num_qk_padded, 2 * head_dim, model_dim)" in gpt
    assert "(num_vo_padded, hdim, model_dim)" in gpt and "# 48" in gpt and "# 16" in gpt
    trainer = _src("train_gpt.py")
    assert "head_dim=128" in trainer and "model_dim=768" in trainer and "num_heads=6" in trainer
    assert bench.BANK_SHAPES == {"qk_bank": (48, 256, 768), "vo_bank": (16, 768, 768), "mlp_bank": (24, 2816, 768)}
    assert [bench.chunk_shape(b) for b in bench.BANK_SHAPES] == [(6, 256, 768), (2, 768, 768), (3, 2816, 768)]
    assert len(MAPS) == 6 and MARGIN == 1.05 and bench.SPLIT_MIN_ROWS == 1024
    cfg = _src("track_1_short", "config.py")
    assert "int(os.environ.get(\"NUM_SCHEDULED_ITERATIONS\", \"1122\"))" in cfg and "SCHEDULE_GROWTH_STEPS = 52" in cfg
    assert bench.RUN_STEPS == 1122 + 52 + 20


def test_new_triton_kernel_parses_and_mirrors_ba_plus_cAA():
    """No Triton on the laptop: check the source parses and keeps the record kernel's structure (the GPU test runs it)."""
    source = _src("track_1_short", "perf", "kernels", "anvil_chain.py")
    tree = ast.parse(source)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert names == {"anvil_aS_plus_SB_kernel", "aS_plus_SB"}
    for needle in ("if m_idx + BLOCK_SIZE_M <= n_idx:", "accumulator = tl.dot(s, bt, accumulator)",
                   "accumulator *= alpha", "accumulator += s_add * beta", "tl.store(c_ptrs_t, output.T, mask=c_mask_t)"):
        assert needle in source, needle


# ------------------------------------------------------------------------------------------------ GPU (the node)

@needs_cuda
def test_gpu_symmetric_small_product_matches_baddbmm():
    dev = torch.device("cuda")
    for label in bench.BANK_SHAPES:
        X = bench.synthetic(bench.chunk_shape(label), "p05", dev, seed=0)
        ops = AK.ops_for(X)
        tall = X.size(-2) > X.size(-1)
        A = ops.gram(X, tall)
        A = (A.float() / AK._frobenius_scale(A, MARGIN, EPS).square()).bfloat16()
        a, b, c = MAPS[0]
        B = ops.poly(A, b, c)
        dense, sym = ops.small(A, B, a), AK.ops_for(X, True).small(A, B, a)
        assert bench.errors(sym, dense)["max_rel_err"] < 2e-2
        assert bench.errors(sym, sym.mT)["max_rel_err"] < 2e-2


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
def test_gpu_kernels_match_cpu_reference(kernel):
    groups, sym = config.ANVIL_KERNELS[kernel]
    for label in bench.BANK_SHAPES:
        X = bench.synthetic(bench.chunk_shape(label), "p05", torch.device("cuda"), seed=1)
        exact = bench.exact_cascade(X)
        base = AK.reference_cascade(X, AK.ops_for(X), MAPS, MARGIN, EPS, X.size(-2) > bench.SPLIT_MIN_ROWS)
        alt = AK.gram_cascade(X, AK.ops_for(X, sym), MAPS, groups, MARGIN, EPS)
        cpu = AK.gram_cascade(X.cpu(), AK.RefOps(sym), MAPS, groups, MARGIN, EPS)
        base_exact = bench.fro_rel(base, exact)
        assert bench.fro_rel(alt, exact) < 3.0 * base_exact + 2e-3
        assert bench.errors(alt.cpu(), cpu)["fro_rel_err"] < 2.0 * base_exact + 0.02


@needs_cuda
@pytest.mark.parametrize("flag", [dict(kernel=k) for k in KERNELS] + [dict(ablate="skip_matrix")])
def test_gpu_bank_update_captures_in_a_cuda_graph(flag):
    """What AnvilBankGraphs does: warm, capture, then replay == eager bit for bit from the same state."""
    from track_1_short.perf.cuda_graphs.capture_support import bits_equal, restore, snapshot
    X = bench.synthetic(bench.chunk_shape("mlp_bank"), "p05", torch.device("cuda"), seed=2)
    bank = bench.make_bank("mlp_bank", X)
    update = bench.flagged_update(bank, **flag)
    with torch.no_grad():
        for _ in range(3):
            update()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            update()
        torch.cuda.synchronize()
        before = snapshot(bank.mutated())
        update()
        eager = snapshot(bank.mutated())
        restore(bank.mutated(), before)
        graph.replay()
        torch.cuda.synchronize()
    assert all(bits_equal(t, eager[name]) for name, t in bank.mutated().items())
    assert not bits_equal(bank.p_slice, before["p_slice"])


def test_gram_space_order_equals_the_record_order_in_float64():
    """In exact arithmetic the Gram-space evaluation (pairs of maps composed on the Gram side) is the record's
    cascade: checked in float64, where rounding is negligible, on a tall matrix with the real maps."""
    torch.manual_seed(0)
    X = torch.randn(96, 24, dtype=torch.float64)
    X = X / (1.05 * torch.linalg.matrix_norm(X))
    eye = torch.eye(24, dtype=torch.float64)
    R = X.clone()
    for a, b, c in MAPS:
        A = R.mT @ R
        R = R @ (a * eye + b * A + c * A @ A)
    G = X.clone()
    for g in range(3):
        A = G.mT @ G
        Q = None
        for j in range(2):
            a, b, c = MAPS[2 * g + j]
            P = a * eye + b * A + c * A @ A
            if j == 0:
                A = P @ A @ P
            Q = P if Q is None else Q @ P
        G = G @ Q
    assert torch.linalg.matrix_norm(G - R) / torch.linalg.matrix_norm(R) < 1e-12
