# Simulator UT — hand-written L1 kernels

Cycle-accurate A5 simulator tests for the **affinity operators**
(`al.copy_from_gm_to_l1_nd`, `al.copy_from_l1_to_ub`, `al.sync_block_set/wait`).

These are pytest, not lit: they need CANN and the Ascend950 simulator, and each takes
10–15 minutes. They are deliberately **outside** the fast lit gate.

## Why they exist

The L1 prefetch pass emits the GM→L1→UB schedule automatically. These kernels build the
same schedule *by hand*, with the same operators, and they are the performance ceiling
the pass is measured against. They are also the only thing that would catch a regression
in the affinity operators themselves.

| Kernel | Schedule | Baseline µs | Manual L1 µs |
|--------|----------|------------:|-------------:|
| GELU `(256,2048)` | dual veccore, even/odd row tasks | 116.36 | **67.97** |
| SELU `(512,512)` | dual veccore, even/odd | 48.81 | **34.07** |
| SwiGLU `(256,2048)` | dual veccore, even/odd | 98.50 | **70.45** |
| RMSNorm `32×8192` | dual veccore, even/odd | 60.29 | **39.40** |
| Softmax `16×4096` | dual veccore, even/odd | 13.24 | **10.07** |

The two `*_baseline.py` kernels are the *inputs the compiler pass transforms*, run at
**1 outer × 8 inner** trips — the geometry every recorded compiler number was measured at
(GELU 116.36 → 68.36 µs, SwiGLU 49.58 → 39.91 µs).

## Running

Each kernel is a self-contained script, run inside the simulator container. They are
**not** driven by pytest: everything under `unittest/` inherits a session fixture that
binds a real NPU device, and there is no device here -- the simulator is the device.

The command must go through `msprof op simulator` -- that is what runs the kernel on the
cycle simulator and emits the timings. Without it the test tries to talk to a real device
and dies with `init soc version failed`.

```bash
cd docker-sim
sudo env DOCKER_IT=0 USE_HOST_BISHENGIR=1 CANN_VER=9.1.0 ./run.sh msprof op simulator \
    env ROWS=256 COLS=2048 BLOCK_M=32 BLOCK_N=2048 PROGRAMS=1 TRITON_FORCE_MULTIBUFFER=0 \
        L1_EARLY_UB_RELEASE=0 L1_UB_DEALLOC=1 BISHENGIR_DISABLE_MEMREF_DSE=1 VF_MERGE_LEVEL=1 \
    python3 /workspace/submodules/triton-ascend/third_party/ascend/unittest/simulator_ut/test_gelu_l1_prefetch_2vec_evenodd.py
```

The baselines (`test_*_baseline.py`) are the inputs the compiler pass transforms; run them
with `OUTER_TRIPS=1 INNER_TRIPS=8 PROGRAMS=1 TRITON_L1_CACHE_PASS=1 (historical legacy option; removed)`.

## The one rule that matters

`bl.dealloc` / `hivm.ub_plan_release` is a PlanMemory lifetime hint — **not** a runtime
free, fence, or DMA-completion dependency. A UB buffer released after an asynchronous
`tl.store` must be handed back with `PIPE_MTE3 → PIPE_MTE1`, never `PIPE_V → PIPE_MTE1`:
the release has to come from the *final hardware reader*, not the last source statement.
Getting this wrong lets the cube's MTE1 refill a private UB while MTE3 is still draining
it, and the output silently contains the *next* input tile.
