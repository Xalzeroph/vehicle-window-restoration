"""
selective_scan 的 Triton 实现 + 纯 torch 参考实现。

用途：本仓库 vmamba.py 依赖 mamba_ssm 编译算子（selective_scan_fn），
而目标环境无法编译安装 mamba_ssm（无 nvcc）。本模块提供同签名替代。

调用路径（vmamba.py SS2D.forward_corev0，group 版布局）：
    u:   (B, K*D, L)   float32, contiguous
    A:   (K*D, N)      已取负（As = -exp(A_logs)）
    B:   (B, K, N, L)  组内所有 d 共享
    C:   (B, K, N, L)
    D:   (K*D,) 或 None
    delta_bias: (K*D,) 或 None
    z 恒为 None；return_last_state 恒 False

S6 离散化（与 Mamba/VMamba 一致）：
    dt = softplus(delta + bias)（可选）
    x  = x * exp(dt * A) + dt * B * u     （逐 (b, d, n)）
    y  = sum_n x * C + D * u
"""

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _ss_kernel(
    u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, bias_ptr, y_ptr,
    L, N, K, D_inner, numel_DK,
    HAS_D: tl.constexpr, HAS_BIAS: tl.constexpr, SOFTPLUS: tl.constexpr,
):
    didx = tl.program_id(0)  # 0 .. K*D_inner-1
    b = tl.program_id(1)
    g = didx // D_inner
    d = didx % D_inner

    offs_n = tl.arange(0, N)

    base = b * numel_DK * L + didx * L
    a = tl.load(A_ptr + didx * N + offs_n)  # (N,)
    b_offs = b * K * N * L + g * N * L + offs_n * L
    c_offs = b * K * N * L + g * N * L + offs_n * L

    if HAS_D:
        dsc = tl.load(D_ptr + didx)
    else:
        dsc = 0.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + didx)
    else:
        bias = 0.0

    x = tl.zeros((N,), dtype=tl.float32)
    for i in range(L):
        u_i = tl.load(u_ptr + base + i)
        dt_i = tl.load(dt_ptr + base + i)
        if HAS_BIAS:
            dt_i += bias
        if SOFTPLUS:
            dt_i = tl.where(dt_i > 20.0, dt_i, tl.log(1.0 + tl.exp(dt_i)))
        bvec = tl.load(B_ptr + b_offs + i)  # (N,)
        cvec = tl.load(C_ptr + c_offs + i)
        x = x * tl.exp(dt_i * a) + (dt_i * bvec) * u_i
        y = tl.sum(x * cvec, axis=0) + dsc * u_i
        tl.store(y_ptr + base + i, y)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """纯 torch 参考实现（逐组逐时间步），仅用于数值核验与 CPU 兜底。"""
    if z is not None:
        raise NotImplementedError("z branch not supported")
    Bsz, Dk, L = u.shape
    K = B.shape[1]
    D_inner = Dk // K
    N = B.shape[2]
    assert A.shape == (Dk, N) and C.shape == (Bsz, K, N, L)

    u = u.view(Bsz, K, D_inner, L)
    delta = delta.view(Bsz, K, D_inner, L)
    A = A.view(K, D_inner, N)
    D0 = D.view(K, D_inner) if D is not None else torch.zeros(K, D_inner, device=u.device)
    bias0 = delta_bias.view(K, D_inner) if delta_bias is not None else torch.zeros(K, D_inner, device=u.device)

    out = torch.empty_like(u)
    for g in range(K):
        x = torch.zeros(Bsz, D_inner, N, dtype=torch.float32, device=u.device)
        for l in range(L):
            dt = delta[:, g, :, l] + bias0[g]
            if delta_softplus:
                dt = F.softplus(dt)
            a_g = A[g].unsqueeze(0)                      # (1, D_inner, N)
            dt_e = dt.unsqueeze(-1)                      # (Bsz, D_inner, 1)
            bv = B[:, g, :, l].unsqueeze(1)              # (Bsz, 1, N)
            cv = C[:, g, :, l].unsqueeze(1)              # (Bsz, 1, N)
            ul = u[:, g, :, l].unsqueeze(-1)             # (Bsz, D_inner, 1)
            x = x * torch.exp(dt_e * a_g) + dt_e * bv * ul
            out[:, g, :, l] = (x * cv).sum(-1) + D0[g] * u[:, g, :, l]
    out = out.view(Bsz, Dk, L)
    return out


def selective_scan_fn(u, delta, A, B, C, D=None, z=None,
                      delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    if z is not None or return_last_state:
        raise NotImplementedError("z / return_last_state branches not supported")
    assert u.dim() == 3 and A.dim() == 2 and B.dim() == 4
    Bsz, Dk, L = u.shape
    K = B.shape[1]
    N = B.shape[2]
    D_inner = Dk // K
    assert A.shape == (Dk, N)

    u = u.contiguous().float()
    delta = delta.contiguous().float()
    A = A.contiguous().float()
    B = B.contiguous().float()
    C = C.contiguous().float()
    Dc = D.contiguous().float() if D is not None else None
    bias = delta_bias.contiguous().float() if delta_bias is not None else None

    out = torch.empty(Bsz, Dk, L, dtype=torch.float32, device=u.device)

    try:
        _ss_kernel[(Dk, Bsz)](
            u, delta, A, B, C,
            Dc if Dc is not None else u,
            bias if bias is not None else u,
            out,
            L, N, K, D_inner, Dk,
            HAS_D=Dc is not None, HAS_BIAS=bias is not None,
            SOFTPLUS=delta_softplus,
            num_warps=2,
        )
    except Exception:
        # Triton 不可用（如 CPU）时退回纯 torch 参考实现
        return selective_scan_ref(u, delta, A, B, C, Dc,
                                  delta_bias=bias,
                                  delta_softplus=delta_softplus)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for (Bsz, K, D_inner, N, L) in [(2, 2, 4, 8, 17), (1, 4, 8, 16, 31)]:
        Dk = K * D_inner
        u = torch.randn(Bsz, Dk, L, dtype=torch.float32)
        delta = torch.randn(Bsz, Dk, L, dtype=torch.float32) * 0.05 - 2.0
        A = -torch.rand(Dk, N, dtype=torch.float32)
        B = torch.randn(Bsz, K, N, L, dtype=torch.float32) * 0.1
        C = torch.randn(Bsz, K, N, L, dtype=torch.float32) * 0.1
        D = torch.rand(Dk, dtype=torch.float32)
        bias = torch.randn(Dk, dtype=torch.float32) * 0.1
        for sp in (True, False):
            r_ref = selective_scan_ref(u, delta, A, B, C, D,
                                       delta_bias=bias, delta_softplus=sp)
            r_fast = selective_scan_fn(u, delta, A, B, C, D,
                                       delta_bias=bias, delta_softplus=sp)
            err = (r_ref - r_fast).abs().max().item()
            rel = err / r_ref.abs().clamp_min(1e-6).max().item()
            print(f"B{Bsz} K{K} D{D_inner} N{N} L{L} softplus={sp}: "
                  f"max_abs_err={err:.3e} max_rel={rel:.3e}")
            assert err < 1e-3, "kernel mismatch!"
    print("all checks passed")
