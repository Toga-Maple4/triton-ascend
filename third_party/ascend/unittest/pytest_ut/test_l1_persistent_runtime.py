"""Dynamic Persistent launch premises, compiled against both public ABIs."""
import ctypes
from copy import deepcopy
import subprocess
import sysconfig

import pytest
from triton.backends.ascend.l1_launch_contract import (
    emit_runtime_guards, validate_runtime_contract, physical_block_count,
)

@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def contract():
    return {"launch": {"distribution": "half_grid", "allow_odd_grid": False},
            "guard": {"kind": "persistent_tiles", "tile_mode": "full",
                      "tile": {"rows": 16, "cols": 2048, "element_bits": 32},
                      "roles": {"inputs": [{"arg": 2, "type": "*fp32"}],
                                "output": {"arg": 3, "type": "*fp32"},
                                "logical_programs": {"arg": 4, "type": "i32"},
                                "persistent_steps": {"arg": 5, "type": "i32"},
                                "row_tasks": {"arg": 6, "type": "i32"},
                                "stride": {"const": 8192, "type": "i32"}}}}


@pytest.mark.parametrize("grid_source", [False, True])
def test_both_compiled_apis(tmp_path, capfd, grid_source):
    c = contract()
    if grid_source:
        c["guard"]["roles"]["logical_programs"] = {"grid": "x", "type": "i32"}
    # A specialized source argument sits between the pointers and the counts.
    signature = {0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32", 4: "i32", 5: "i32"}
    definition, py, exported = emit_runtime_guards(c, {2: 1}, signature, 32)
    assert physical_block_count(c) == "gridX / 2"
    assert "persistent_grid_count_mismatch" in py and "persistent_grid_count_mismatch" in exported
    source = tmp_path / "guard.cpp"
    source.write_text('#include <Python.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n' + definition +
                     '\nextern "C" PyObject* py(int gridX, int gridY, int gridZ, int32_t _arg3, int32_t _arg4, int32_t _arg5) {\n' + py + '\nPy_RETURN_TRUE; }\n' +
                     'extern "C" int check(int gridX, int gridY, int gridZ, int num_args, const void** kernel_args, const size_t* arg_sizes) {\nint dispatched = 0; auto launch = [&]() {\n' + exported + '\ndispatched = 1; }; launch(); return dispatched; }\n')
    so = tmp_path / "guard.so"
    subprocess.run(["c++", "-shared", "-fPIC", "-I" + sysconfig.get_path("include"), str(source), "-o", str(so)], check=True)
    lib = ctypes.PyDLL(str(so))
    lib.py.argtypes, lib.py.restype = [ctypes.c_int] * 6, ctypes.py_object
    lib.check.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    lib.check.restype = ctypes.c_int
    cases = [(4, 1, 1, 4, 8, 6, True), (4, 1, 1, 4, 1, 100, True),
             (4, 1, 1, 4, 7, 6, True), (4, 1, 1, 4, 9, 1, True),
             (3, 1, 1, 3, 8, 6, False), (0, 1, 1, 0, 8, 6, False),
             (66, 1, 1, 66, 8, 6, False), (4, 2, 1, 4, 8, 6, False),
             (4, 1, 1, 4, 0, 6, False), (4, 1, 1, 4, 8, -1, False),
             (4, 1, 1, 4, 2147483647, 1, True),
             (6, 1, 1, 6, 2147483647, 1, False),
             (4, 1, 1, 4, 134217728, 134217728, True),
             (4, 1, 1, 4, 134217732, 134217729, False)]
    if not grid_source:
        cases.append((4, 1, 1, 6, 8, 6, False))
    for gx, gy, gz, n, steps, tasks, valid in cases:
        scalars = [ctypes.c_int32(v) for v in (n, steps, tasks)]
        pointers = (ctypes.c_void_p * 5)(1, 2, *[ctypes.addressof(v) for v in scalars])
        sizes = (ctypes.c_size_t * 5)(8, 8, 4, 4, 4)
        assert bool(lib.check(gx, gy, gz, 5, pointers, sizes)) == valid
        if valid:
            assert lib.py(gx, gy, gz, n, steps, tasks) is True
        else:
            with pytest.raises(ValueError, match="persistent_"):
                lib.py(gx, gy, gz, n, steps, tasks)
    if not grid_source:
        assert "persistent_grid_count_mismatch" in capfd.readouterr().err


@pytest.mark.parametrize("mutation", [
    lambda c: c["launch"].update(distribution="resident_pairs", resident_stride=28),
    lambda c: c["launch"].update(logical_programs=4),
    lambda c: c["launch"].update(logical_grid="xyz"),
    lambda c: c["launch"].update(allow_odd_grid=True),
    lambda c: c["guard"].update(tile_mode="partial"),
    lambda c: c["guard"]["roles"]["logical_programs"].update(grid="x"),
    lambda c: c["guard"]["roles"].update(logical_programs={"grid": "w", "type": "i32"}),
    lambda c: c["guard"]["roles"].update(stride={"arg": 4, "type": "i32"}),
])
def test_malformed_interface(mutation):
    c = contract()
    mutation(c)
    with pytest.raises(RuntimeError):
        validate_runtime_contract(c)
