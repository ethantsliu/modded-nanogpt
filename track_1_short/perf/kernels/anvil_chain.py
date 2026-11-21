"""Symmetric two-operand product for the Gram-space ANVIL cascade (perf/anvil_kernel.py, ANVIL_KERNEL=gram*s):
C = beta * S + alpha * (S @ B), for symmetric S and B that commute.

What it replaces: torch.baddbmm(S, S, B, beta=beta, alpha=alpha) on the small (Gram-sized) matrices of the
Gram-space cascade, plus the defensive copy baddbmm makes when its input and batch1 are the same tensor.
Why it is faster: S and B are polynomials in the same Gram matrix, so they commute and S @ B is symmetric; like the
kernels in polar_express.py, only the blocks on one side of the diagonal are computed and each is mirrored into the
other triangle. It is ba_plus_cAA_kernel with the second factor read from B instead of A (B[k, n] is loaded as the
row B[n, k], so both loads are coalesced).
Invariants: S and B are square, symmetric and commute (they are not checked); C is written in full. Within a
diagonal block the two triangles are separate dot products (equal in exact arithmetic, a rounding apart in bf16).

Off the record path: nothing imports this module unless ANVIL_KERNEL names an "s" kernel (config.py).
Never compiled or run when written (no GPU, no Triton on the authoring machine): anvil-kernel/bench.py and
test_optim_kernel.py check it against baddbmm on the node.
"""
import torch
import triton
import triton.language as tl

from track_1_short.perf.kernels.polar_express import _pid_to_block


@triton.jit
def anvil_aS_plus_SB_kernel(
    S_ptr, B_ptr, C_ptr,
    M,
    s_stride_b, s_stride_r, s_stride_c,
    b_stride_b, b_stride_r, b_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # Only the blocks on and below the diagonal are computed (polar_express.py's LOWER_UPPER=1); the rest are mirrored
    if m_idx + BLOCK_SIZE_M <= n_idx:
        return

    S_ptr += batch_idx * s_stride_b
    B_ptr += batch_idx * b_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # C[m, n] = sum_k S[m, k] * B[k, n] = sum_k S[m, k] * B[n, k]  (B symmetric): two row loads
    s_ptrs = S_ptr + (offs_m[:, None] * s_stride_r + offs_k[None, :] * s_stride_c)
    bt_ptrs = B_ptr + (offs_n[:, None] * b_stride_r + offs_k[None, :] * b_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        k_remaining = M - k * BLOCK_SIZE_K
        s = tl.load(s_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        bt_temp = tl.load(bt_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        bt = tl.trans(bt_temp)
        accumulator = tl.dot(s, bt, accumulator)
        s_ptrs += BLOCK_SIZE_K * s_stride_c
        bt_ptrs += BLOCK_SIZE_K * b_stride_c

    # The block of S to add (the current block of C)
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    s_add_ptrs = S_ptr + (offs_cm[:, None] * s_stride_r + offs_cn[None, :] * s_stride_c)
    s_add = tl.load(s_add_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    accumulator *= alpha
    accumulator += s_add * beta

    output = accumulator.to(C_ptr.dtype.element_ty)

    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    tl.store(c_ptrs, output, mask=c_mask)

    # The block mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def aS_plus_SB(S: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """out = beta * S + alpha * (S @ B) for symmetric, commuting S and B ([M, M] or [batch, M, M]); out is neither."""
    assert S.ndim in (2, 3) and S.shape == B.shape == out.shape and S.size(-1) == S.size(-2)
    assert out.data_ptr() != S.data_ptr() and out.data_ptr() != B.data_ptr()
    M = S.size(-1)
    batch_size = S.size(0) if S.ndim == 3 else 1
    batch_stride = lambda t: t.stride(0) if t.ndim == 3 else 0

    # ba_plus_cAA's tile config (H100 autotuning at M=768)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    anvil_aS_plus_SB_kernel[grid](
        S_ptr=S, B_ptr=B, C_ptr=out,
        M=M,
        s_stride_b=batch_stride(S), s_stride_r=S.stride(-2), s_stride_c=S.stride(-1),
        b_stride_b=batch_stride(B), b_stride_r=B.stride(-2), b_stride_c=B.stride(-1),
        c_stride_b=batch_stride(out), c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
        alpha=alpha, beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        num_stages=4, num_warps=8,
    )
    return out
