"""Shared utilities for running Triton kernels on the A5 NPU simulator."""

import torch


def sim_zeros_like(t):
    return torch.zeros(t.shape, dtype=t.dtype).to(t.device)

def sim_empty_like(t):
    return torch.empty(t.shape, dtype=t.dtype).to(t.device)

def sim_zeros(shape, dtype, device):
    return torch.zeros(shape, dtype=dtype).to(device)

def sim_randn(shape, dtype, device="npu"):
    return torch.randn(shape, dtype=dtype).to(device)

def sim_rand(shape, dtype, device="npu"):
    return torch.rand(shape, dtype=dtype).to(device)

def sim_ones(shape, dtype, device="npu"):
    return torch.ones(shape, dtype=dtype).to(device)

def sim_full(shape, fill_value, dtype, device="npu"):
    return torch.full(shape, fill_value, dtype=dtype).to(device)

def sim_randint(low, high, shape, dtype=torch.int64, device="npu"):
    return torch.randint(low, high, shape, dtype=dtype).to(device)

def assert_close(name, actual, expected, atol=1e-2, rtol=1e-2):
    a = actual.cpu().float()
    e = expected.cpu().float()
    max_err = (a - e).abs().max().item()
    ok = torch.allclose(a, e, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_abs_err={max_err:.6f} (atol={atol}, rtol={rtol})")
    assert ok, f"{name} mismatch: max_abs_err={max_err}"


_COMPILE_ONLY = False

def _sim_do_bench(fn, warmup=25, rep=100, grad_to_none=None,
                  quantiles=None, return_mode="mean"):
    fn()
    if quantiles:
        return [1.0] * len(quantiles)
    return 1.0


def patch_triton_autotune_for_simulator():
    try:
        import triton.backends.ascend.driver as ascend_driver
        _orig_clear = ascend_driver.NPUDriver.clear_cache
        def _safe_clear_cache(self, cache):
            try:
                _orig_clear(self, cache)
            except RuntimeError:
                pass
        ascend_driver.NPUDriver.clear_cache = _safe_clear_cache
        ascend_driver.NPUDriver.get_benchmarker = lambda self: _sim_do_bench
    except (ImportError, AttributeError):
        pass

    try:
        import triton.runtime.jit as jit_mod
        _orig_jit_run = jit_mod.JITFunction.run
        def _compile_only_run(self, *args, grid, warmup=False, **kwargs):
            if _COMPILE_ONLY:
                return _orig_jit_run(self, *args, grid=grid, warmup=True, **kwargs)
            return _orig_jit_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        jit_mod.JITFunction.run = _compile_only_run
    except (ImportError, AttributeError):
        pass

    try:
        import triton.testing as testing
        testing.do_bench = _sim_do_bench
    except (ImportError, AttributeError):
        pass

    try:
        import triton.runtime.autotuner as autotuner_mod
        autotuner_mod.Autotuner.do_bench = staticmethod(_sim_do_bench)

        import re as _re
        from triton.compiler.errors import MLIRCompilationError, CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources

        _orig_bench = autotuner_mod.Autotuner._bench
        def _logging_bench(self, *args, config, **meta):
            kernel_name = getattr(self, 'base_fn', self.fn).__name__ if hasattr(self, 'base_fn') else '?'
            config_str = str(config)
            current = dict(meta, **config.all_kwargs())
            full_nargs = {**self.nargs, **current}

            def kernel_call():
                if config.pre_hook:
                    config.pre_hook(full_nargs)
                self.pre_hook(full_nargs)
                try:
                    self.fn.run(*args, **current)
                except Exception as e:
                    try:
                        self.post_hook(full_nargs, exception=e)
                    finally:
                        raise
                self.post_hook(full_nargs, exception=None)

            try:
                result = self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
                print(f"  [CONFIG] {kernel_name} | {config_str} => OK")
                return result
            except (OutOfResources, CompileTimeAssertionFailure, MLIRCompilationError) as e:
                err_str = str(e)
                overflow_m = _re.search(
                    r"ub overflow,\s*requires\s+(\d+)\s+bits\s+while\s+(\d+)\s+bits\s+available",
                    err_str,
                )
                if overflow_m:
                    print(f"  [CONFIG] {kernel_name} | {config_str} => COMPILE_FAIL (ub_overflow: {overflow_m.group(1)} / {overflow_m.group(2)} bits)")
                else:
                    errors = _re.findall(r"error:\s*(.+?)(?:\n|$)", err_str)
                    key_errors = [e for e in errors if "ub overflow" in e or "Failed to run" in e or "failed to legalize" in e]
                    if key_errors:
                        detail = key_errors[0].strip()[:100]
                    elif errors:
                        detail = errors[0].strip()[:100]
                    else:
                        detail = err_str[:100].replace('\n', ' ')
                    print(f"  [CONFIG] {kernel_name} | {config_str} => COMPILE_FAIL ({detail})")
                return [float("inf"), float("inf"), float("inf")]
            except Exception as e:
                err_str = str(e)[:100].replace('\n', ' ')
                print(f"  [CONFIG] {kernel_name} | {config_str} => ERROR ({err_str})")
                return [float("inf"), float("inf"), float("inf")]
        autotuner_mod.Autotuner._bench = _logging_bench
    except (ImportError, AttributeError):
        pass
