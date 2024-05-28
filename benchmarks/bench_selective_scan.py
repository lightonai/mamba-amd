import torch
import pytest

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def time_(fn, name, steps=500):
    steps = 500
    # warm up
    for _ in range(100):
        fn()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    print(f'Time {name}: {sum(times) / steps} ms')


def bench_selective_scan(is_variable_B, is_variable_C, varBC_groups, has_D, has_z, has_delta_bias,
                        delta_softplus, return_last_state, seqlen, itype, wtype):
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    dim = 5376
    dstate = 16
    is_complex = wtype == torch.complex64
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype,
                    requires_grad=True)
    if not is_variable_C:
        C_shape = (dim, dstate)
    elif varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        C_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype,
                    requires_grad=True)
    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
    if has_z:
        z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    else:
        z = None
    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
    else:
        delta_bias = None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()

    ss = lambda: selective_scan_fn(
            u, delta, A, B, C, D, z=z,
            delta_bias=delta_bias, delta_softplus=delta_softplus,
            return_last_state=return_last_state
        )
    time_(ss, f'[{seqlen}] selective scan fwd', steps=100)

    for _ in range(30):
        out, *rest = selective_scan_fn(
            u, delta, A, B, C, D, z=z,
            delta_bias=delta_bias, delta_softplus=delta_softplus,
            return_last_state=return_last_state
        )
        g = torch.randn_like(out)
        out.backward(g)

    steps = 50
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        out, *rest = selective_scan_fn(
            u, delta, A, B, C, D, z=z,
            delta_bias=delta_bias, delta_softplus=delta_softplus,
            return_last_state=return_last_state
        )
        g = torch.randn_like(out)
        start_events[i].record()
        out.backward(g)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    print(f'Time [{seqlen}] selective scan bwd:', sum(times) / steps)


bench_selective_scan(True, True, 2, True, True, True, True, True, 128, torch.float16, torch.float32)
bench_selective_scan(True, True, 2, True, True, True, True, True, 256, torch.float16, torch.float32)
bench_selective_scan(True, True, 2, True, True, True, True, True, 512, torch.float16, torch.float32)
bench_selective_scan(True, True, 2, True, True, True, True, True, 1024, torch.float16, torch.float32)
bench_selective_scan(True, True, 2, True, True, True, True, True, 2048, torch.float16, torch.float32)
bench_selective_scan(True, True, 2, True, True, True, True, True, 4096, torch.float16, torch.float32)

