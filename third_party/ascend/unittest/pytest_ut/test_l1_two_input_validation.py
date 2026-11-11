"""Runtime launch contract regression checks, including both compiled ABIs."""
import ctypes
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from triton.backends.ascend.l1_launch_contract import (
    SCHEMA_VERSION, validate_runtime_contract, validate_compile_result,
    emit_runtime_guards, validate_launch_metadata, emit_launch_guards,
)
from l1_contract_fixtures import (
    CONTRACT, STATIC_CONTRACT, STRIDED_CONTRACT, JAGGED_CONTRACT,
    FLAT_CONTRACT, FIXED_CONTRACT, TWO_INPUT_CONTRACT, arg, const, change,
)

@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield

CONTRACT = TWO_INPUT_CONTRACT
SIGNATURE = {0:"*fp16",1:"*fp16",2:"*fp16",3:"i32",4:"i32",5:"i32"}

def test_generated_guard_mapping_and_signature():
    signature = {0: "*fp16", 1: "*fp16", 2: "*fp16", 3: "i32",
                 4: "i32", 5: "i32", 6: "i32"}
    _, python, c_api = emit_runtime_guards(CONTRACT, {3: 99}, signature, 32)
    assert "_arg5, _arg6, _arg4" in python
    for index in (3, 4, 5):
        assert f"arg_sizes[{index}] != sizeof(int32_t)" in c_api
    assert "num_args != 6" in c_api
    # Unused arguments are not an alias or role-completeness obligation.
    emit_runtime_guards(CONTRACT, {}, {**SIGNATURE, 6: "*i8"}, 32)


@pytest.mark.parametrize("pointers", [
    ("*fp32", "*fp16", "*fp32"),
    ("*fp16", "*bf16", "*fp16"),
])
def test_mismatched_two_input_signatures_are_rejected(pointers):
    signature = {**SIGNATURE, **dict(enumerate(pointers))}
    with pytest.raises(RuntimeError, match="signature"):
        emit_runtime_guards(CONTRACT, {}, signature, 32)


def test_compiled_predicate(tmp_path):
    predicate, _, _ = emit_runtime_guards(CONTRACT, {}, SIGNATURE, 32)
    source = tmp_path / "guard.cpp"
    source.write_text("#include <cstdint>\n" + predicate +
                      '\nextern "C" bool check(int n,int y,int z,int64_t r,int64_t c,int64_t s) {'
                      "return valid_runtime_program_split(n,y,z,r,c,s);}\n")
    shared = tmp_path / "guard.so"
    subprocess.run(["c++", "-shared", "-fPIC", str(source), "-o", str(shared)], check=True)
    check = ctypes.CDLL(str(shared)).check
    check.argtypes = [ctypes.c_int] * 3 + [ctypes.c_int64] * 3
    check.restype = ctypes.c_bool
    for shape in ((4, 128, 8192, 8192), (4, 80, 2048, 2048), (64, 16, 2048, 4096)):
        n, rows, cols, stride = shape
        assert check(n, 1, 1, rows, cols, stride)
    for shape in ((3, 128, 8192, 8192), (66, 128, 8192, 8192),
                  (4, 127, 8192, 8192), (4, 128, 8191, 8192),
                  (4, 128, 8192, 2048), (4, 128, 8192, 1 << 31),
                  (4, 1 << 31, 8192, 8192), (1 << 30, 128, 8192, 8192)):
        n, rows, cols, stride = shape
        assert not check(n, 1, 1, rows, cols, stride)
    assert not check(4, 2, 1, 128, 8192, 8192)


@pytest.mark.parametrize("pointer,bits,columns", [("*fp16",16,2048), ("*bf16",16,2048), ("*fp32",32,1024), ("*fp32",32,512)])
def test_two_input_guards_compile_and_agree_across_abis(tmp_path, pointer, bits, columns):
    """Exercise the emitted guards, including compact C argument positions."""
    import sysconfig
    signature = {0: pointer, 1: pointer, 2: pointer, 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}
    definition, python_guard, c_guard = emit_runtime_guards(
        change(CONTRACT, pointer=pointer, cols=columns), {3: 16}, signature, 32)
    source = tmp_path / 'strided.cpp'
    source.write_text(
        '#include <Python.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n' + definition +
        '\nextern "C" PyObject* py_check(int gridX,int gridY,int gridZ,'
        'int32_t _arg4,int32_t _arg5,int32_t _arg6) {\n' + python_guard + '\nPy_RETURN_TRUE;\n}\n' +
        'extern "C" int c_check(int gridX,int gridY,int gridZ,int num_args,'
        'const void **kernel_args,const size_t *arg_sizes) {\nint dispatched=0;\n'
        'auto launch=[&]() {\n' + c_guard + '\ndispatched=1;\n};\nlaunch();\nreturn dispatched;\n}\n')
    shared = tmp_path / 'strided.so'
    subprocess.run(['c++', '-shared', '-fPIC', '-I' + sysconfig.get_path('include'),
                    str(source), '-o', str(shared)], check=True)
    library = ctypes.PyDLL(str(shared))
    py = library.py_check
    py.argtypes, py.restype = [ctypes.c_int] * 6, ctypes.py_object
    c = library.c_check
    c.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    c.restype = ctypes.c_int
    for grid, stride, rows, cols, accepted in [
        ((4,1,1),8192,96,8192,True),
        ((4,1,1),8256,96,8192,True),
        ((4,1,1),8192,16,2048,True),  # inactive logical workers
        ((4,1,1),8191,96,8192,False),
        ((4,1,1),0,96,8192,False),
        ((4,1,1),-8192,96,8192,False),
        ((4,1,1),8192,97,8192,False),
        ((4,1,1),8192,96,8191,False),
        ((3,1,1),8192,96,8192,False),
        ((66,1,1),8192,96,8192,False),
        ((4,2,1),8192,96,8192,False),
        ((4,1,1),8192,2147483632,8192,False),
        ((4,1,1),2147483647,2147483632,2147481600,False),
    ]:
        if accepted:
            assert py(*grid, stride, rows, cols) is True
        else:
            with pytest.raises(ValueError):
                py(*grid, stride, rows, cols)
        scalars = [ctypes.c_int32(v) for v in (stride, rows, cols)]
        pointers = (ctypes.c_void_p * 6)(1, 2, 3, *[ctypes.addressof(v) for v in scalars])
        sizes = (ctypes.c_size_t * 6)(8, 8, 8, 4, 4, 4)
        assert c(*grid, 6, pointers, sizes) == accepted
    assert c(4,1,1,6,None,None) == 0
    sizes[3] = 8
    assert c(4,1,1,6,pointers,sizes) == 0
