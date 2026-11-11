"""SwiGLU forward plain-UB baseline for manual L1 comparisons.

The JIT helper and forward kernel below are copied verbatim from the current
root kernels/test_swiglu.py. Only this harness is environment-driven.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from harness_config import load_swiglu_config
from utils import (  # noqa: E402
    assert_close,
    patch_triton_autotune_for_simulator,
    sim_empty_like,
)

patch_triton_autotune_for_simulator()


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_fwd_kernel(
    a,
    b,
    c,
    stride_row,
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

            a_ptrs = a + rows_off[:, None] * stride_row + cols_off[None, :]
            b_ptrs = b + rows_off[:, None] * stride_row + cols_off[None, :]
            c_ptrs = c + rows_off[:, None] * stride_row + cols_off[None, :]

            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0)

            a_f32 = a_chunk.to(tl.float32)
            silu_a = silu(a_f32)

            c_chunk = silu_a.to(a_chunk.dtype) * b_chunk

            tl.store(c_ptrs, c_chunk, mask=block_mask)


def test_swiglu_baseline():
    config = load_swiglu_config()
    shape = (config.rows, config.cols)
    dtype = torch.float16
    multibuffer = os.environ.get("TRITON_FORCE_MULTIBUFFER", "<unset>")
    print(
        "[swiglu-baseline] "
        f"shape={shape} BLOCK_M={config.block_m} "
        f"BLOCK_N={config.block_n} PROGRAMS={config.programs} "
        f"TRITON_FORCE_MULTIBUFFER={multibuffer}"
    )

    a = torch.randn(shape, dtype=dtype).npu()
    b = torch.randn(shape, dtype=dtype).npu()
    c = sim_empty_like(a)

    _swiglu_fwd_kernel[(config.programs,)](
        a,
        b,
        c,
        a.stride(0),
        config.rows,
        config.cols,
        BLOCK_SIZE_N=config.block_n,
        BLOCK_SIZE_M=config.block_m,
        debug=True,
        l1_pointer_args_noalias=True,
    )

    a_f32 = a.cpu().float()
    b_f32 = b.cpu().float()
    silu_a_ref = a_f32 * torch.sigmoid(a_f32)
    ref_c = silu_a_ref * b_f32
    assert_close("swiglu_fwd_baseline", c.cpu(), ref_c.half())


if __name__ == "__main__":
    test_swiglu_baseline()
