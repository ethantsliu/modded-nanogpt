"""Alternative implementations of the ANVIL whitening cascade (ANVIL_KERNEL) and its measurement-only ablation
(ANVIL_ABLATE). Off by default; see anvil-kernel/bench.py.

What it replaces: anvil_cascade (optim/anvil.py) for the banks in ANVIL_BANKS. The record runs, per map k of the
six ANVIL_MAPS (a, b, c):  A = Gram(X);  B = b*A + c*A@A;  X = a*X + X@B  -- one Gram and one large GEMM per map.

The Gram-space cascade (same update rule): with P_k = a*I + B_k, a map is X <- X @ P_k, and P_k is a polynomial in
the Gram A_k, so A_{k+1} = P_k A_k P_k needs no access to X. A group of g consecutive maps is therefore evaluated on
the small Gram-sized matrices only (A, B, and the accumulated Q = P_k ... P_{k+g-1}), and X is touched twice per
group: one Gram at its start, one GEMM X <- X @ Q at its end. For mlp_bank (2816x768) the groups (2, 2, 2) run 3
Grams and 3 large GEMMs instead of 6 and 6, and drop the two full-size elementwise passes of the normalization and
the six of the split a*X + X@B; they add three 768x768 products per group. The Frobenius normalization is folded in:
the first group's Q is divided by d, so X itself is never rescaled.

Why groups, not all six maps at once: the Gram is stored in bf16, so its small eigenvalues are rounding noise, and
composing more than three maps on it without re-reading X amplifies that noise (measured: 4+ chained maps diverge on
heavy-tailed spectra; groups of 2 or 3 stay at the record's own distance from the exact-arithmetic result).

Why it might not be faster: every kernel here is under one wave of CTAs on an H100, so latency, not FLOPs, sets the
cost; and the bank updates may be hidden under comms anyway. anvil-kernel/bench.py measures the first, the
ANVIL_ABLATE=skip_matrix arm the second.

Invariants: same inputs and outputs as anvil_cascade (reads bank.grad, updates bank.velocity in place, returns a
fresh bf16 tensor the caller may overwrite); no host sync, so it captures in the optimizer CUDA graphs; the rails
pre-pass is anvil_cascade's own text, compiled the same way. This module imports neither Triton nor optim/anvil.py
at import time (the CPU tests and the bench's CPU smoke run it through RefOps).
"""
import torch
from torch import Tensor

from track_1_short.config import ANVIL_ABLATE, ANVIL_BANKS, ANVIL_KERNEL, ANVIL_KERNELS


def configure(kernel: str = "", ablate: str = "", banks=()):
    """Set the flags after import (anvil-kernel/bench.py and the tests; the trainer reads them from config.py)."""
    global ANVIL_KERNEL, ANVIL_ABLATE, ANVIL_BANKS
    assert kernel == "" or kernel in ANVIL_KERNELS, kernel
    assert ablate in ("", "skip_matrix") and not (kernel and ablate)
    ANVIL_KERNEL, ANVIL_ABLATE = kernel, ablate
    ANVIL_BANKS = frozenset(banks) if kernel or ablate else frozenset()


def applies(label: str) -> bool:
    """Does a flag reroute this bank's cascade? False for every bank in the record configuration (ANVIL_BANKS is
    empty unless a flag is set)."""
    return label in ANVIL_BANKS


def describe() -> str:
    """One line for the log (printed only when a flag is set)."""
    what = f"ANVIL_KERNEL={ANVIL_KERNEL}" if ANVIL_KERNEL else f"ANVIL_ABLATE={ANVIL_ABLATE}"
    note = (" (same update rule, Gram-space evaluation)" if ANVIL_KERNEL
            else " -- MEASUREMENT ONLY, NOT A VALID TRAINING RUN: the whitening cascade is skipped")
    return f"[anvil-kernel] {what} banks={sorted(ANVIL_BANKS)}{note}"


# -----------------------------------------------------------------------------
# The matrix primitives, on two backends

class RefOps:
    """Plain torch on any device: every product is accumulated in fp32 and stored in bf16, which is what the Triton
    kernels and cuBLAS do. The CPU reference of both the record's cascade and the alternatives."""

    def __init__(self, sym_small: bool = False):
        self.sym_small = sym_small

    def gram(self, X: Tensor, tall: bool) -> Tensor:
        Xf = X.float()
        return (Xf.mT @ Xf if tall else Xf @ Xf.mT).bfloat16()

    def poly(self, A: Tensor, b: float, c: float) -> Tensor:
        Af = A.float()
        return (b * Af + c * (Af @ Af)).bfloat16()

    def small(self, S: Tensor, B: Tensor, a: float) -> Tensor:
        Sf = S.float()
        out = a * Sf + Sf @ B.float()
        if self.sym_small:  # the symmetric kernel computes one triangle and mirrors it
            out = torch.tril(out) + torch.tril(out, -1).mT
        return out.bfloat16()

    def apply(self, X: Tensor, Q: Tensor, tall: bool) -> Tensor:
        return (X.float() @ Q.float() if tall else Q.float() @ X.float()).bfloat16()

    def update(self, X: Tensor, B: Tensor, a: float, tall: bool, split: bool) -> Tensor:
        Xf = X.float()
        return (a * Xf + (Xf @ B.float() if tall else B.float() @ Xf)).bfloat16()


class CudaOps:
    """The record's kernels (perf/kernels/polar_express.py, torch.bmm / baddbmm), launched eagerly; `sym_small`
    swaps the small baddbmm for the symmetric two-operand kernel (perf/kernels/anvil_chain.py)."""

    def __init__(self, sym_small: bool = False):
        from track_1_short.perf.kernels import polar_express
        self.k = polar_express
        self.sym_small = sym_small
        if sym_small:
            from track_1_short.perf.kernels.anvil_chain import aS_plus_SB
            self.aS_plus_SB = aS_plus_SB

    def gram(self, X: Tensor, tall: bool) -> Tensor:
        s = X.size(-1) if tall else X.size(-2)
        A = torch.empty((*X.shape[:-2], s, s), device=X.device, dtype=X.dtype)
        return self.k.XTX(X, out=A) if tall else self.k.XXT(X, out=A)

    def poly(self, A: Tensor, b: float, c: float) -> Tensor:
        return self.k.ba_plus_cAA(A, alpha=c, beta=b, out=torch.empty_like(A))

    def small(self, S: Tensor, B: Tensor, a: float) -> Tensor:
        if self.sym_small:
            return self.aS_plus_SB(S, B, alpha=1.0, beta=a, out=torch.empty_like(S))
        return (torch.baddbmm if S.ndim > 2 else torch.addmm)(S, S, B, beta=a)

    def apply(self, X: Tensor, Q: Tensor, tall: bool) -> Tensor:
        return torch.matmul(X, Q) if tall else torch.matmul(Q, X)

    def update(self, X: Tensor, B: Tensor, a: float, tall: bool, split: bool) -> Tensor:
        """anvil_cascade's a*X + X@B (tall) / a*X + B@X (wide), split into two kernels for the large banks."""
        if split:
            C = torch.matmul(X, B) if tall else torch.matmul(B, X)
            return C.add_(X, alpha=a)
        addmm = torch.baddbmm if X.ndim > 2 else torch.addmm
        return addmm(X, X, B, beta=a) if tall else addmm(X, B, X, beta=a)


_OPS: dict[tuple[bool, bool], object] = {}


def ops_for(X: Tensor, sym_small: bool = False):
    key = (X.is_cuda, sym_small)
    if key not in _OPS:
        _OPS[key] = CudaOps(sym_small) if X.is_cuda else RefOps(sym_small)
    return _OPS[key]


# -----------------------------------------------------------------------------
# The cascade: the record's evaluation order (a reference for the bench and the tests) and the Gram-space one

def _frobenius_scale(A: Tensor, margin: float, eps: float) -> Tensor:
    """d = margin * ||X||_F + eps from the unnormalized Gram's trace, [..., 1, 1] fp32 (anvil_cascade's)."""
    tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
    return tr.sqrt() * margin + eps


def reference_cascade(X: Tensor, ops, maps, margin: float, eps: float, split: bool = False) -> Tensor:
    """anvil_cascade's matrix part (from the bf16 momentum X on), kernel for kernel, as eager calls on `ops`."""
    tall = X.size(-2) > X.size(-1)
    A = ops.gram(X, tall)
    d = _frobenius_scale(A, margin, eps)
    X = (X.float() / d).bfloat16()
    A = (A.float() / d.square()).bfloat16()
    for k, (a, b, c) in enumerate(maps):
        if k > 0:
            A = ops.gram(X, tall)
        X = ops.update(X, ops.poly(A, b, c), a, tall, split)
    return X


def gram_cascade(X: Tensor, ops, maps, groups: tuple[int, ...], margin: float, eps: float) -> Tensor:
    """The same maps as reference_cascade, each group composed in Gram space (module docstring). X: bf16
    [..., rows, cols], unnormalized; returns a new tensor."""
    assert sum(groups) == len(maps) and min(groups) >= 1
    tall = X.size(-2) > X.size(-1)
    k = 0
    for group_index, size in enumerate(groups):
        A = ops.gram(X, tall)
        if group_index == 0:
            d = _frobenius_scale(A, margin, eps)
            A = (A.float() / d.square()).bfloat16()
        Q = None
        for j in range(size):
            a, b, c = maps[k]
            k += 1
            B = ops.poly(A, b, c)                      # B = b*A + c*A@A, so this map is X <- X @ (a*I + B)
            if j < size - 1:                           # the next map's Gram: (a*I + B) A (a*I + B)
                A = ops.small(ops.small(A, B, a), B, a)
            if Q is None:
                B.diagonal(dim1=-2, dim2=-1).add_(a)   # Q = a*I + B (B is not read again)
                Q = B
            else:
                Q = ops.small(Q, B, a)                 # Q <- Q @ (a*I + B)
        if group_index == 0:
            Q = (Q.float() / d).bfloat16()             # the normalization X / d, applied to Q instead
        X = ops.apply(X, Q, tall)
    return X


# -----------------------------------------------------------------------------
# The rails pre-pass (anvil_cascade's first lines, verbatim) and the two entry points

def _rails_body(grad_chunk: Tensor, velocity: Tensor, momentum_t: Tensor, fast_beta_t: Tensor,
                fast_weight_t: Tensor, slow_beta: float) -> Tensor:
    """Twin-rail Nesterov momentum: updates velocity in place, returns the bf16 lookahead blend."""
    grad_chunk = grad_chunk.float()
    momentum = momentum_t.to(grad_chunk.dtype)
    velocity[0].lerp_(grad_chunk, 1 - fast_beta_t.to(grad_chunk.dtype))
    velocity[1].lerp_(grad_chunk, 1 - slow_beta)
    w = fast_weight_t.to(grad_chunk.dtype)
    blend = w * velocity[0] + (1 - w) * velocity[1]
    g = grad_chunk.lerp_(blend, momentum)
    return g.bfloat16().contiguous()


def _rails_normalized_body(grad_chunk: Tensor, velocity: Tensor, momentum_t: Tensor, fast_beta_t: Tensor,
                           fast_weight_t: Tensor, slow_beta: float, margin: float, eps: float) -> Tensor:
    """ANVIL_ABLATE=skip_matrix: the rails, then only the Frobenius normalization (the cascade with zero maps).
    One reduction and one elementwise pass stand in for the cascade's own normalization pass."""
    grad_chunk = grad_chunk.float()
    momentum = momentum_t.to(grad_chunk.dtype)
    velocity[0].lerp_(grad_chunk, 1 - fast_beta_t.to(grad_chunk.dtype))
    velocity[1].lerp_(grad_chunk, 1 - slow_beta)
    w = fast_weight_t.to(grad_chunk.dtype)
    blend = w * velocity[0] + (1 - w) * velocity[1]
    g = grad_chunk.lerp_(blend, momentum)
    X = g.bfloat16().float()  # the cascade normalizes the bf16-rounded momentum
    d = X.square().sum(dim=(-2, -1), keepdim=True).sqrt() * margin + eps
    return (X / d).bfloat16()


_rails_compiled = torch.compile(_rails_body, dynamic=False, fullgraph=True)
_rails_normalized_compiled = torch.compile(_rails_normalized_body, dynamic=False, fullgraph=True)


def rails(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, slow_beta: float) -> Tensor:
    """Compiled on the GPU (as in anvil_cascade), the eager body on the CPU (no CPU inductor compile)."""
    fn = _rails_compiled if grad_chunk.is_cuda else _rails_body
    return fn(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, slow_beta)


def skip_matrix(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, slow_beta: float,
                margin: float, eps: float) -> Tensor:
    fn = _rails_normalized_compiled if grad_chunk.is_cuda else _rails_normalized_body
    return fn(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, slow_beta, margin, eps)


def bank_cascade(bank, maps, slow_beta: float, margin: float, eps: float) -> Tensor:
    """anvil_cascade's replacement for one AnvilBank under ANVIL_KERNEL / ANVIL_ABLATE (optim/anvil.py passes its
    own ANVIL_MAPS, RAIL_SLOW_BETA, FROBENIUS_MARGIN and FROBENIUS_EPS)."""
    state = (bank.grad, bank.velocity, bank.momentum, bank.fast_beta, bank.fast_weight)
    if ANVIL_ABLATE == "skip_matrix":
        return skip_matrix(*state, slow_beta, margin, eps)
    groups, sym_small = ANVIL_KERNELS[ANVIL_KERNEL]
    X = rails(*state, slow_beta)
    return gram_cascade(X, ops_for(X, sym_small), maps, groups, margin, eps)
