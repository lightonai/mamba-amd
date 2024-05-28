import torch
import torch.nn as nn


from mamba_ssm.ops.triton.selective_state_update import selective_state_update

from mamba_ssm.ops.triton.layernorm import layer_norm_fn
from .bench_selective_scan import time_

def run_benchmarks(dim, dstate, has_z, itype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None

    ssu = lambda: selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    time_(ssu, 'selective state update', steps=100)

    steps = 100
    state = torch.randn((batch_size, dim, dstate), device='cuda', dtype=itype)
    norm = nn.LayerNorm(state.shape[-1])
    norm.to(device='cuda')
    residual = torch.randn_like(state)

    lnf = lambda: layer_norm_fn(
            state,
            norm.weight,
            norm.bias,
            residual=residual,
            prenorm=True,
            residual_in_fp32=False,
            eps=norm.eps,
        )
    time_(lnf, 'layer norm forward', steps=100)

    for _ in range(30):
        out, res = layer_norm_fn(
            state,
            norm.weight,
            norm.bias,
            residual=residual,
            prenorm=True,
            residual_in_fp32=False,
            eps=norm.eps,
        )
        g = torch.randn_like(out)
        out.backward(g)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        out, res = layer_norm_fn(
            state,
            norm.weight,
            norm.bias,
            residual=residual,
            prenorm=True,
            residual_in_fp32=False,
            eps=norm.eps,
        )
        g = torch.randn_like(out)
        start_events[i].record()
        out.backward(g)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    print(f'Time tridao layer norm bwd:', sum(times) / steps)


run_benchmarks(4096, 2688, False, torch.float16)

