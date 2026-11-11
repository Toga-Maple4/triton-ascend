"""GELU forward — plain UB baseline, pinned to the L1 comparison config.

This is the apples-to-apples baseline for `test_gelu_l1.py` /
`test_gelu_l1_prefetch.py`: SAME shape/block, 1 AI core (grid=(1,) via PROGRAMS=1),
multibuffer controlled by TRITON_FORCE_MULTIBUFFER (run with =0 for the comparison).

The kernel body `_gelu_fwd_kernel` is copied VERBATIM from test_gelu.py (R1: identical
compute) — the only thing this file does is fix the harness to the comparison config
and run fwd only (no bwd), so the per-core timing lines up directly with the L1
variants. At BM=32,BN=2048 a single program loads a full [32,2048] f16 tile (128KB)
and casts it to f32 (256KB temp) on ONE veccore -> exceeds the 248KB UB, so this is
expected to OVERFLOW (that overflow is the whole motivation for the L1 staging: the
L1 variants split the tile into per-veccore [16,2048] halves that DO fit).

Run:
  TRITON_FORCE_MULTIBUFFER=0 PROGRAMS=1 ROWS=256 COLS=2048 BLOCK_M=32 BLOCK_N=2048 \
    scripts/run_pvk_test.sh kernels/gelu/test_gelu_baseline.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from utils import patch_triton_autotune_for_simulator, sim_empty_like, assert_close  # noqa: E402

patch_triton_autotune_for_simulator()

# 1 program == 1 AI core, to isolate the GM->UB cost vs the L1 MIX variant (grid=(1,)).
NUM_PROGRAMS = int(os.environ.get("PROGRAMS", "1") or "1")


@triton.jit
def gelu_tanh_approx(x):
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    return 0.5 * x * (1 + tl.math.tanh(tanh_arg))


# VERBATIM from test_gelu.py::_gelu_fwd_kernel (do not modify the kernel body).
@triton.jit
def _gelu_fwd_kernel(
    x,
    y,
    stride_row: tl.constexpr,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            y_ptrs = y + rows_off[:, None] * stride_row + cols_off[None, :]

            x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)

            x_f32 = x_chunk.to(tl.float32)
            y_f32 = gelu_tanh_approx(x_f32)

            y_chunk = y_f32.to(x_chunk.dtype)

            tl.store(y_ptrs, y_chunk, mask=block_mask)


def _envint(name, default):
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


def _resolved_extent(name, default, trip_name, block_size):
    trips = os.environ.get(trip_name, "").strip()
    if not trips:
        return _envint(name, default)
    trip_count = int(trips)
    if trip_count <= 0:
        raise ValueError(f"{trip_name} must be positive, got {trip_count}")
    return trip_count * block_size


def test_gelu_baseline():
    print("=== GELU Forward — plain UB baseline (pinned comparison config) ===")
    block_m = _envint("BLOCK_M", 32)
    block_n = _envint("BLOCK_N", 2048)
    n_rows = _resolved_extent("ROWS", 256, "OUTER_TRIPS", block_m)
    n_cols = _resolved_extent("COLS", 2048, "INNER_TRIPS", block_n)
    shape = (n_rows, n_cols)
    dtype = torch.float16
    mb = os.environ.get("TRITON_FORCE_MULTIBUFFER", "<unset>")
    print(f"[gelu-baseline] shape={shape} BLOCK_M={block_m} BLOCK_N={block_n} "
          f"PROGRAMS={NUM_PROGRAMS} TRITON_FORCE_MULTIBUFFER={mb}")

    x = torch.randn(shape, dtype=dtype).npu()
    y = sim_empty_like(x)

    _gelu_fwd_kernel[(NUM_PROGRAMS,)](
        x, y,
        x.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE_N=block_n, BLOCK_SIZE_M=block_m,
        debug=True,
        l1_pointer_args_noalias=True,
    )

    x_f32 = x.cpu().float()
    sqrt_2_over_pi = 0.7978845608028654
    ref_y = 0.5 * x_f32 * (1.0 + torch.tanh(sqrt_2_over_pi * (x_f32 + 0.044715 * x_f32 ** 3)))
    assert_close("gelu_fwd_baseline", y.cpu(), ref_y.half())

    print("=== GELU baseline: PASSED ===")


if __name__ == "__main__":
    test_gelu_baseline()
