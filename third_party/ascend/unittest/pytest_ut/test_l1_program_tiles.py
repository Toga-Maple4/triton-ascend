"""Host validation of whole-program XYZ distribution and per-input strides."""
import ctypes
import importlib.util
from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

# This contract-only module needs neither a Triton extension nor a device.
path = Path(__file__).resolve().parents[2] / "backend/l1_launch_contract.py"
spec = importlib.util.spec_from_file_location("program_tiles_contract", path)
contract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract_module)


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def contract():
    arg = lambda index, ty: {"arg": index, "type": ty}
    return {
        "launch": {"distribution": "resident_pairs", "resident_stride": 28,
                   "allow_odd_grid": True, "logical_grid": "xyz"},
        "guard": {"kind": "program_tiles", "tile_mode": "partial",
                  "tile": {"rows": 256, "cols": 128, "element_bits": 16},
                  "roles": {"inputs": [arg(2, "*bf16"), arg(3, "*bf16")],
                            "output": arg(4, "*fp32"),
                            "input_strides": [arg(5, "i32"), arg(6, "i32")]}}}


def test_program_grid_and_stride_predicate(tmp_path):
    c = contract()
    definition, _, _ = contract_module.emit_runtime_guards(
        c, {}, {0: "*bf16", 1: "*bf16", 2: "*fp32", 3: "i32", 4: "i32"}, 28)
    src, shared = tmp_path / "guard.cpp", tmp_path / "guard.so"
    src.write_text('#include <cstdint>\n#include <algorithm>\n' + definition +
                   '\nextern "C" bool valid(int x,int y,int z,int64_t a,int64_t b) {'
                   'return valid_program_tiles(x,y,z,a,b);}\n' +
                   'extern "C" int blocks(int gridX,int gridY,int gridZ) { return ' +
                   contract_module.physical_block_count(c) + ';}\n')
    subprocess.run(["c++", "-shared", "-fPIC", str(src), "-o", str(shared)], check=True)
    library = ctypes.CDLL(str(shared))
    valid = library.valid
    valid.argtypes = [ctypes.c_int] * 3 + [ctypes.c_int64] * 2
    valid.restype = ctypes.c_bool
    for grid in ((1, 1, 1), (4, 128, 1), (3, 5, 7), (57, 1, 1)):
        assert valid(*grid, 128, 131)
        assert library.blocks(*grid) == min((grid[0] * grid[1] * grid[2] + 1) // 2, 28)
    for grid in ((0, 1, 1), (1, -1, 1), (1, 1, 0), (1 << 30, 2, 1),
                 (1 << 30, 1, 2), ((1 << 31) - 1, (1 << 31) - 1, 2)):
        assert not valid(*grid, 128, 128)
    assert not valid(4, 128, 1, 127, 128)
    assert not valid(4, 128, 1, 128, 1 << 31)


def test_stride_roles_use_specialized_abi_and_constants():
    c = contract()
    c["guard"]["roles"]["input_strides"][1] = {"const": 128, "type": "i32"}
    signature = {0: "*bf16", 1: "i32", 2: "*bf16", 3: "*fp32", 4: "i32"}
    _, python, exported = contract_module.emit_runtime_guards(c, {1: 1}, signature, 28)
    assert "_arg4, 128" in python
    assert "kernel_args[3]" in exported
    assert "num_args != 4" in exported


@pytest.mark.parametrize("change", [
    lambda c: c["launch"].update(logical_grid="x"),
    lambda c: c["launch"].update(allow_odd_grid=False),
    lambda c: c["guard"]["roles"].update(input_strides=[]),
    lambda c: c["guard"]["roles"]["input_strides"][0].update(const=128),
    lambda c: c["guard"]["roles"]["input_strides"][0].update(type="i64"),
    lambda c: c["guard"]["roles"].update(output={"arg": 2, "type": "*bf16"}),
])
def test_malformed_program_contract_rejected(change):
    c = deepcopy(contract())
    change(c)
    with pytest.raises(RuntimeError):
        contract_module.validate_runtime_contract(c)
