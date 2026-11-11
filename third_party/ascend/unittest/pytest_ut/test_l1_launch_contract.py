"""Runtime launch contract regression checks, including both compiled ABIs."""
import ctypes
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import triton.language as tl
from triton.backends.ascend.l1_launch_contract import (
    SCHEMA_VERSION, validate_runtime_contract, validate_compile_result,
    emit_runtime_guards, validate_launch_metadata, emit_launch_guards,
)
from l1_contract_fixtures import (
    CONTRACT, STATIC_CONTRACT, STRIDED_CONTRACT, JAGGED_CONTRACT,
    FLAT_CONTRACT, FIXED_CONTRACT, TWO_INPUT_CONTRACT, arg, const, change,
)

def driver_target():
    from triton.backends.compiler import GPUTarget
    return GPUTarget('ascend', 'Ascend950PR_9589', 0)


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def test_contract_schema_changes_the_full_compilation_cache_key(tmp_path):
    """A schema change misses a populated cache and reruns the compiler pipeline."""
    import triton.backends.ascend.l1_launch_contract as contract

    # Simulate the bytes read after a wheel upgrade without writing any installed
    # file. Fresh processes also match the lifetime of triton_key's memoization.
    script = textwrap.dedent("""\
        import builtins, importlib, io, json, os, sys
        from pathlib import Path
        from types import SimpleNamespace
        import triton.backends.ascend as backend
        path = Path(sys.argv[1]).resolve()
        backend.__path__ = [str(path.parent)]
        compiler = importlib.import_module('triton.compiler.compiler')
        original = path.read_bytes()
        marker = ('SCHEMA_VERSION = ' + sys.argv[2]).encode()
        assert original.count(marker) == 1
        changed = original.replace(marker, ('SCHEMA_VERSION = ' + str(int(sys.argv[2]) + 1)).encode())
        real_open = builtins.open
        intercepted = []
        def read_schema(file, mode='r', *args, **kwargs):
            if mode == 'rb' and Path(file).resolve() == path:
                intercepted.append(str(path))
                return io.BytesIO(changed if sys.argv[3] == 'changed' else original)
            return real_open(file, mode, *args, **kwargs)
        builtins.open = read_schema
        fixed = SimpleNamespace(hash=lambda: 'unchanged')
        key = compiler.get_cache_key(fixed, fixed, fixed, {})
        assert intercepted, 'backend contract was not included in triton_key'
        # Exercise compile() and the real file/group cache. Only the expensive
        # backend stage and device-handle construction are replaced by host fakes.
        os.environ['TRITON_CACHE_DIR'] = sys.argv[4]
        os.environ['TRITON_ALWAYS_COMPILE'] = '0'
        version = int(sys.argv[2]) + (sys.argv[3] == 'changed')
        calls = []
        class Options:
            def hash(self): return 'unchanged'
        class Backend:
            def hash(self): return 'unchanged'
            def parse_options(self, options): return Options()
            def load_dialects(self, context): pass
            def get_codegen_implementation(self, options): return {}
            def get_module_map(self): return {}
            def add_stages(self, stages, options, language):
                def emit(module, metadata):
                    calls.append('compile')
                    metadata['l1_program_split_schema'] = version
                    return b'host-fixture-binary'
                stages['ttir'] = emit
        compiler.make_backend = lambda target: Backend()
        compiler.CompiledKernel = lambda src, group, key: SimpleNamespace(
            key=key, metadata=json.loads(Path(group['cached.json']).read_text()))
        compiler.ir = SimpleNamespace(context=lambda: SimpleNamespace(disable_multithreading=lambda: None),
                                      load_dialects=lambda context: None)
        compiler.buffer_ir = compiler.ascend_ir = SimpleNamespace(load_dialects=lambda context: None)
        src = compiler.ASTSource.__new__(compiler.ASTSource)
        src.name, src.ext = 'cached', 'ttir'
        src.language = compiler.Language.TRITON
        src.hash = lambda: 'unchanged'
        src.parse_options = lambda: {}
        src.make_ir = lambda *args: 'host fixture'
        target = compiler.GPUTarget('ascend', 'Ascend950PR_9589', 0)
        result = compiler.compile(src, target=target, _env_vars={})
        print(json.dumps(dict(full_key=key, cache_key=result.key,
                             compiled=len(calls), schema=result.metadata['l1_program_split_schema'])))
    """)

    def cache_key(version):
        result = subprocess.run(
            [sys.executable, '-c', script, contract.__file__, str(SCHEMA_VERSION), version, str(tmp_path / 'cache')],
            check=True, capture_output=True, text=True)
        return json.loads(result.stdout.splitlines()[-1])

    before = cache_key('original')
    assert before['compiled'] == 1 and before['schema'] == SCHEMA_VERSION
    hit = cache_key('original')
    assert hit == {**before, 'compiled': 0}
    changed = cache_key('changed')
    assert changed['compiled'] == 1 and changed['schema'] == SCHEMA_VERSION + 1
    assert changed['cache_key'] != before['cache_key']
    assert changed['full_key'] != before['full_key']
    assert cache_key('changed') == {**changed, 'compiled': 0}
    assert cache_key('original') == hit
    assert len(list((tmp_path / 'cache').glob('*/__grp__cached.json'))) == 2


def test_both_launch_guards_and_constexpr_argument_mapping():
    definition, python, c_api = emit_runtime_guards(
        CONTRACT, {2: 8192, 5: 2048, 6: 32},
        {0: "*fp16", 1: "*fp16", 2: "i32", 3: "i32", 4: "i32", 5: "i32", 6: "i32"}, 32)
    assert "_arg3, _arg4" in python
    assert "kernel_args[2]" in c_api and "kernel_args[3]" in c_api
    assert "sizeof(int32_t)" in c_api and "num_args != 4" in c_api
    assert "valid_runtime_program_split" in definition
    with pytest.raises(RuntimeError):
        emit_runtime_guards(CONTRACT, {}, {0: "*fp32"}, 32)


def test_compiled_argument_guards_use_python_names_and_compact_c_abi(tmp_path):
    import sysconfig
    definition, python_guard, c_guard = emit_runtime_guards(
        CONTRACT, {2: 8192, 5: 2048, 6: 32},
        {0: "*fp16", 1: "*fp16", 2: "i32", 3: "i32", 4: "i32", 5: "i32", 6: "i32"}, 32)
    source = tmp_path / "guards.cpp"
    source.write_text(
        '#include <Python.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n' +
        definition +
        '\nextern "C" PyObject* python_check(int gridX, int gridY, int gridZ,'
        'int32_t _arg3, int32_t _arg4) {\n' + python_guard + '\nPy_RETURN_TRUE;\n}\n' +
        'extern "C" int c_check(int gridX, int gridY, int gridZ, int num_args,'
        'const void **kernel_args, const size_t *arg_sizes) {\nint dispatched = 0;\n'
        'auto launch = [&]() {\n' + c_guard + '\ndispatched = 1;\n};\nlaunch();\nreturn dispatched;\n}\n')
    shared = tmp_path / "guards.so"
    subprocess.run(["c++", "-shared", "-fPIC", "-I" + sysconfig.get_path("include"),
                    str(source), "-o", str(shared)], check=True)
    library = ctypes.PyDLL(str(shared))
    python = library.python_check
    python.argtypes = [ctypes.c_int] * 5
    python.restype = ctypes.py_object
    assert python(4, 1, 1, 128, 8192) is True
    for args in ((4, 1, 1, 129, 8192), (66, 1, 1, 128, 8192), (4, 2, 1, 128, 8192)):
        with pytest.raises(ValueError, match="runtime contract or physical core limit"):
            python(*args)
    c_api = library.c_check
    c_api.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    c_api.restype = ctypes.c_int
    rows, cols = ctypes.c_int32(128), ctypes.c_int32(8192)
    pointers = (ctypes.c_void_p * 4)(1, 2, ctypes.addressof(rows), ctypes.addressof(cols))
    sizes = (ctypes.c_size_t * 4)(8, 8, 4, 4)
    assert c_api(4, 1, 1, 4, pointers, sizes) == 1
    assert c_api(66, 1, 1, 4, pointers, sizes) == 0
    rows.value = 129
    assert c_api(4, 1, 1, 4, pointers, sizes) == 0
    rows.value = 128
    sizes[2] = 8
    assert c_api(4, 1, 1, 4, pointers, sizes) == 0
    assert c_api(4, 1, 1, 3, None, None) == 0


@pytest.mark.parametrize("pointer", ["*fp16", "*bf16"])
def test_compiled_static_guards_have_no_runtime_column_argument(tmp_path, pointer):
    import sysconfig
    definition, python_guard, c_guard = emit_runtime_guards(
        change(STATIC_CONTRACT, pointer=pointer), {2: 8192, 4: 8192, 5: 2048, 6: 32},
        {0: pointer, 1: pointer, 2: "i32", 3: "i32", 4: "i32", 5: "i32", 6: "i32"}, 32)
    assert "_arg3, 8192" in python_guard
    assert "kernel_args[3]" not in c_guard
    source = tmp_path / "static_guards.cpp"
    source.write_text(
        '#include <Python.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n' +
        definition +
        '\nextern "C" PyObject* python_check(int gridX,int gridY,int gridZ,int32_t _arg3) {\n' +
        python_guard + '\nPy_RETURN_TRUE;\n}\n' +
        'extern "C" int c_check(int gridX,int gridY,int gridZ,int num_args,'
        'const void **kernel_args,const size_t *arg_sizes) {\nint dispatched=0;\n'
        'auto launch=[&]() {\n' + c_guard + '\ndispatched=1;\n};\nlaunch();\nreturn dispatched;\n}\n')
    shared = tmp_path / "static_guards.so"
    subprocess.run(["c++", "-shared", "-fPIC", "-I" + sysconfig.get_path("include"),
                    str(source), "-o", str(shared)], check=True)
    library = ctypes.PyDLL(str(shared))
    python = library.python_check
    python.argtypes = [ctypes.c_int] * 4
    python.restype = ctypes.py_object
    assert python(4, 1, 1, 160) is True
    for args in ((4,1,1,129), (66,1,1,128), (4,2,1,128), (0,1,1,128)):
        with pytest.raises(ValueError):
            python(*args)
    c_api = library.c_check
    c_api.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    c_api.restype = ctypes.c_int
    rows = ctypes.c_int32(160)
    pointers = (ctypes.c_void_p * 3)(1, 2, ctypes.addressof(rows))
    sizes = (ctypes.c_size_t * 3)(8, 8, 4)
    assert c_api(4, 1, 1, 3, pointers, sizes) == 1
    assert c_api(66, 1, 1, 3, pointers, sizes) == 0
    rows.value = 129
    assert c_api(4, 1, 1, 3, pointers, sizes) == 0
    rows.value = 160
    sizes[2] = 8
    assert c_api(4, 1, 1, 3, pointers, sizes) == 0
    assert c_api(4, 1, 1, 4, None, None) == 0


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("specialized_value", [1, 16])
def test_fp32_strided_guards_compile_and_agree_across_abis(tmp_path, partial, specialized_value):
    """Exercise the emitted guards, including compact C argument positions."""
    import sysconfig
    signature = {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32', 4: 'i32', 5: 'i32'}
    definition, python_guard, c_guard = emit_runtime_guards(
        change(STRIDED_CONTRACT, mode="partial") if partial else STRIDED_CONTRACT,
        {3: specialized_value}, signature, 32)
    # A source scalar between stride and rows is specialized away. All three
    # surviving shape scalars have the same type, so type checks alone would
    # not catch a source-position/ABI-position mix-up.
    assert '_arg4, _arg5, _arg2' in python_guard
    assert 'memcpy(&split_rows, kernel_args[3]' in c_guard
    assert 'memcpy(&split_cols, kernel_args[4]' in c_guard
    assert 'memcpy(&split_stride, kernel_args[2]' in c_guard
    source = tmp_path / 'strided.cpp'
    source.write_text(
        '#include <Python.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n' + definition +
        '\nextern "C" PyObject* py_check(int gridX,int gridY,int gridZ,'
        'int32_t _arg2,int32_t _arg4,int32_t _arg5) {\n' + python_guard + '\nPy_RETURN_TRUE;\n}\n' +
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
        ((4,1,1),8192,97,8192,partial),
        ((4,1,1),8192,96,8191,partial),
        ((4,1,1),3,1,3,partial),
        ((2,1,1),2051,17,2051,partial),
        ((2,1,1),2051,17,2052,False),
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
        pointers = (ctypes.c_void_p * 5)(1, 2, *[ctypes.addressof(v) for v in scalars])
        sizes = (ctypes.c_size_t * 5)(8, 8, 4, 4, 4)
        assert c(*grid, 5, pointers, sizes) == accepted
    assert c(4,1,1,5,None,None) == 0
    sizes[2] = 8
    assert c(4,1,1,5,pointers,sizes) == 0


def test_specialization_removes_the_scalar_before_backend_argument_numbering():
    import triton
    from triton._C.libtriton import ir
    from triton.compiler.compiler import ASTSource
    from triton.backends.ascend.compiler import NPUOptions

    @triton.jit
    def specialized_shape(x, y, stride, unit, rows, cols):
        value = tl.load(x)
        tl.store(y + stride, value)
        tl.store(y + rows, value)
        tl.store(y + cols, value + unit)

    # This is the constants entry that JIT equal_to_1 specialization supplies,
    # not a tl.constexpr annotation and not a constant-valued contract role.
    source = ASTSource(specialized_shape,
                       dict(x='*fp32', y='*fp32', stride='i32', unit='i32', rows='i32', cols='i32'),
                       constexprs={'unit': 1})
    context = ir.context()
    ir.load_dialects(context)
    module = source.make_ir(driver_target(), NPUOptions(), {}, {}, context)
    function = module.get_function('specialized_shape')
    for index in range(5):
        assert function.args(index) is not None
    with pytest.raises(IndexError):
        function.args(5)
    # The same compact positions used by the compiled guard test above.
    text = str(module)
    for index in (2, 3, 4):
        assert f'tt.addptr %y, %{("stride", "rows", "cols")[index - 2]}' in text


def test_unknown_launch_kind_cannot_dispatch_as_ordinary():
    with pytest.raises(RuntimeError, match="launch kind"):
        validate_launch_metadata(SimpleNamespace(l1_program_split_kind="unknown"), 1)


@pytest.mark.parametrize("pointers", [("*bf16", "*fp16"), ("*fp16", "*bf16"), ("*bf16", "*fp32")])
def test_mixed_bf16_pointer_signatures_are_rejected(pointers):
    with pytest.raises(RuntimeError, match="signature"):
        emit_runtime_guards(CONTRACT, {}, {0: pointers[0], 1: pointers[1], 2: "i32", 3: "i32"}, 32)


@pytest.mark.parametrize("physical_limit", [24, 32])
def test_jagged_guards_compile_and_agree_across_abis(tmp_path, physical_limit):
    import sysconfig
    # A constexpr between runtime arguments must not shift the exported C ABI.
    signature = {0: "*bf16", 1: "*i64", 2: "i32", 3: "*bf16", 4: "constexpr",
                 5: "i32", 6: "i32"}
    definition, python_guard, c_guard = emit_runtime_guards(
        JAGGED_CONTRACT, {4: None}, signature, physical_limit)
    source = tmp_path / 'jagged.cpp'
    source.write_text(
        '#include <Python.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n' + definition +
        '\nextern "C" PyObject* py_check(int gridX,int gridY,int gridZ,'
        'int32_t _arg2,int32_t _arg5,int32_t _arg6) {\n' + python_guard + '\nPy_RETURN_TRUE;\n}\n' +
        'extern "C" int c_check(int gridX,int gridY,int gridZ,int num_args,'
        'const void **kernel_args,const size_t *arg_sizes) {\nint dispatched=0;\n'
        'auto launch=[&]() {\n' + c_guard + '\ndispatched=1;\n};\nlaunch();\nreturn dispatched;\n}\n')
    shared = tmp_path / 'jagged.so'
    subprocess.run(['c++', '-shared', '-fPIC', '-I' + sysconfig.get_path('include'),
                    str(source), '-o', str(shared)], check=True)
    library = ctypes.PyDLL(str(shared))
    py = library.py_check
    py.argtypes, py.restype = [ctypes.c_int] * 6, ctypes.py_object
    c = library.c_check
    c.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    c.restype = ctypes.c_int
    for grid, width, stride, matrix, accepted in [
        ((256,1,1),256,256,128*256,True),
        ((66,1,1),257,257,513*257,True),
        ((2,1,1),17,17,17,True),
        ((3,1,1),256,256,32768,False),
        ((258,1,1),256,256,32768,True),
        ((2048,1,1),256,256,32768,True),
        ((8192,1,1),256,256,32768,True),
        # Large grids are bounded by output indexing, not the old tested cap.
        ((65534,1,1),256,256,32768,True),
        ((65536,1,1),256,256,32768,False),
        (((1 << 31) - 2,1,1),1,1,1,True),
        (((1 << 31) - 2,1,1),1,1,2,False),
        ((8192,1,1),256,256,262144,False),
        ((0,1,1),256,256,32768,False),
        ((2,2,1),256,256,32768,False),
        ((2,1,1),128,256,32768,False),
        ((2,1,1),0,0,32768,False),
        ((2,1,1),256,256,0,False),
        ((2,1,1),256,256,32769,False),
        ((256,1,1),256,256,1<<24,False),
    ]:
        accepted = accepted and min(grid[0] // 2, 32) <= physical_limit
        scalars = [ctypes.c_int32(v) for v in (width, stride, matrix)]
        pointers = (ctypes.c_void_p * 6)(1, 2, ctypes.addressof(scalars[0]), 3,
                                       ctypes.addressof(scalars[1]), ctypes.addressof(scalars[2]))
        sizes = (ctypes.c_size_t * 6)(8,8,4,8,4,4)
        if accepted:
            assert py(*grid, width, stride, matrix)
        else:
            with pytest.raises(ValueError, match="jagged"):
                py(*grid, width, stride, matrix)
        assert bool(c(*grid, 6, pointers, sizes)) == accepted
    assert c(2,1,1,5,pointers,sizes) == 0
    assert c(2,1,1,6,None,sizes) == 0
    pointers[4] = None
    assert c(2,1,1,6,pointers,sizes) == 0


def test_flat_cpp_guard_allows_tails_odd_and_large_grids(tmp_path):
    definition, _, _ = emit_runtime_guards(
        FLAT_CONTRACT, {}, {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 28)
    src = tmp_path / 'flat.cpp'
    src.write_text('#include <cstdint>\n' + definition +
                   '\nextern "C" bool check(int x,int y,int z,int64_t n) {'
                   'return valid_flat_program_split(x,y,z,n);}\n')
    shared = tmp_path / 'flat.so'
    subprocess.run(['c++', '-shared', '-fPIC', str(src), '-o', str(shared)], check=True)
    check = ctypes.CDLL(str(shared)).check
    check.argtypes = [ctypes.c_int] * 3 + [ctypes.c_int64]
    check.restype = ctypes.c_bool
    for grid in (1, 3, 56, 57, 256, 512, 4097):
        for size in (0, 1, 32767, 32768, 32769, 4194304):
            assert check(grid, 1, 1, size)
    for grid, y, z, size in ((0,1,1,1), (512,2,1,1), (512,1,2,1),
                             (512,1,1,-1), (1<<30,1,1,1), (512,1,1,1<<31)):
        assert not check(grid,y,z,size)


def test_flat_compiled_python_and_c_guards(tmp_path):
    import sysconfig
    definition, pyguard, cguard = emit_runtime_guards(
        FLAT_CONTRACT, {4:32768}, {0:'*fp16',1:'*fp16',2:'*fp16',3:'i32',4:'i32'},28)
    src=tmp_path/'flat_guards.cpp'
    src.write_text('#include <Python.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n'+definition+
        '\nextern "C" PyObject* pycheck(int gridX,int gridY,int gridZ,int32_t _arg3) {\n'+pyguard+'\nPy_RETURN_TRUE;\n}\n'+
        'extern "C" int ccheck(int gridX,int gridY,int gridZ,int num_args,const void **kernel_args,const size_t *arg_sizes) {\nint dispatched=0;\nauto call=[&]() {\n'+cguard+'\ndispatched=1;\n};call();return dispatched;\n}\n')
    so=tmp_path/'flat_guards.so'
    subprocess.run(['c++','-shared','-fPIC','-I'+sysconfig.get_path('include'),str(src),'-o',str(so)],check=True)
    lib=ctypes.PyDLL(str(so))
    lib.pycheck.argtypes=[ctypes.c_int]*4;lib.pycheck.restype=ctypes.py_object
    assert lib.pycheck(513,1,1,4194307) is True
    with pytest.raises(ValueError):lib.pycheck(513,2,1,4194307)
    lib.ccheck.argtypes=[ctypes.c_int]*4+[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_size_t)]
    lib.ccheck.restype=ctypes.c_int
    n=ctypes.c_int32(4194307)
    ptrs=(ctypes.c_void_p*4)(1,2,3,ctypes.addressof(n));sizes=(ctypes.c_size_t*4)(8,8,8,4)
    assert lib.ccheck(513,1,1,4,ptrs,sizes)==1
    assert lib.ccheck(513,1,2,4,ptrs,sizes)==0
    assert lib.ccheck(513,1,1,3,None,None)==0
    sizes[3]=8
    assert lib.ccheck(513,1,1,4,ptrs,sizes)==0


def test_effective_compile_options_are_checked():
    from triton.backends.ascend.l1_launch_contract import validate_compile_result
    options = dict(enable_vf_fusion=True, disable_vf_reachable_check=False,
                   enable_auto_multi_buffer=False, hfusion_max_buffer_count_tuning=0)
    result = dict(schema_version=SCHEMA_VERSION, kind="ordinary", reason="ordinary_compiled",
                  compile_options=options)
    validate_compile_result(result, {})
    for update in ({"enable_vf_fusion": 1}, {"hfusion_max_buffer_count_tuning": False},
                   {"unknown": True}):
        with pytest.raises(RuntimeError, match="compile options"):
            validate_compile_result(dict(result, compile_options=dict(options, **update)), {})



@pytest.mark.parametrize('fixture', [CONTRACT, STATIC_CONTRACT, STRIDED_CONTRACT,
                                      JAGGED_CONTRACT, FLAT_CONTRACT, FIXED_CONTRACT,
                                      TWO_INPUT_CONTRACT])
def test_result_metadata_roundtrip_and_cache_hit(tmp_path, fixture):
    from triton.backends.ascend.compiler import _read_program_split_result
    meta = dict(target=SimpleNamespace(arch='Ascend950PR_9589'), mix_mode='aiv')
    result = dict(schema_version=SCHEMA_VERSION, kind='split_mix_half_grid',
                  target=meta['target'].arch, **fixture)
    path = tmp_path / 'kernel.program_split.json'
    path.write_text(json.dumps(result))
    _read_program_split_result(path, meta)
    assert meta['l1_program_split_runtime_contract'] == fixture
    # Cache hits use the copied metadata after the transient output has gone.
    path.unlink()
    cached = json.loads(json.dumps({k: v for k, v in meta.items() if k != 'target'}))
    validate_launch_metadata(SimpleNamespace(**cached), 1)
    assert cached['l1_program_split_schema'] == SCHEMA_VERSION


@pytest.mark.parametrize('content', [None, '{', '{}', '{"schema_version":1}',
    '{"schema_version":2,"kind":"ordinary"}'])
def test_missing_failed_or_malformed_result_does_not_mutate_metadata(tmp_path, content):
    from triton.backends.ascend.compiler import _read_program_split_result
    path = tmp_path / 'failed.program_split.json'
    meta = dict(target=SimpleNamespace(arch='Ascend950PR_9589'), mix_mode='aiv',
                l1_program_split_kind='ordinary', l1_program_split_reason='previous')
    before = deepcopy(meta)
    if content is not None:
        path.write_text(content)
    with pytest.raises(RuntimeError):
        _read_program_split_result(path, meta)
    assert meta == before


def test_concurrent_private_result_directories():
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory
    from triton.backends.ascend.compiler import _read_program_split_result

    def compile_result(index):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.program_split.json'
            result = dict(schema_version=SCHEMA_VERSION, kind='ordinary', reason=str(index))
            path.write_text(json.dumps(result))
            meta = dict(l1_program_split_runtime_contract=CONTRACT)
            _read_program_split_result(path, meta)
        assert not path.exists()
        assert 'l1_program_split_runtime_contract' not in meta
        return meta['l1_program_split_reason']
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(compile_result, range(12))) == list(map(str, range(12)))


@pytest.mark.parametrize('schema', [None, True, 1, 3, '2'])
def test_old_or_malformed_cached_metadata_requires_recompile(schema):
    with pytest.raises(RuntimeError, match='recompile'):
        validate_launch_metadata(SimpleNamespace(l1_program_split_kind='split_mix_half_grid',
            l1_program_split_schema=schema, l1_program_split_runtime_contract=CONTRACT), 1)


@pytest.mark.parametrize('role', [None, {}, {'arg': 4, 'const': 2, 'type': 'i32'},
    {'arg': True, 'type': 'i32'}, {'const': True, 'type': 'i32'},
    {'arg': 1, 'type': 'i32'}, {'const': 1 << 31, 'type': 'i32'},
    {'arg': 4, 'type': 'i64'}, {'arg': 4, 'type': []}])
def test_invalid_scalar_role(role):
    c = deepcopy(CONTRACT)
    c['guard']['roles']['rows'] = role
    with pytest.raises(RuntimeError):
        validate_runtime_contract(c)


@pytest.mark.parametrize('role', [None, {'arg': [], 'type': '*fp16'},
    {'const': 2, 'type': '*fp16'}, {'arg': 2, 'type': []},
    {'arg': 2, 'type': '*bf16'}, {'arg': 3, 'type': '*fp16'}])
def test_invalid_pointer_role(role):
    c = deepcopy(CONTRACT)
    c['guard']['roles']['inputs'] = [role]
    with pytest.raises(RuntimeError):
        validate_runtime_contract(c)


def test_shared_scalar_reordered_roles_unused_arguments_and_constants():
    # Interface-level coverage independent of the compiled workload corpus.
    c = deepcopy(CONTRACT)
    c['guard']['roles'] = dict(inputs=[arg(5, '*fp16')], output=arg(3, '*fp16'),
                               rows=arg(6), cols=arg(4), stride=arg(4))
    sig = {0:'*i8', 1:'*fp16', 2:'i32', 3:'*fp16', 4:'i32', 5:'fp32'}
    _, python, exported = emit_runtime_guards(c, {}, sig, 32)
    assert '_arg4, _arg2, _arg2' in python
    assert exported.count('arg_sizes[2]') == 1
    c['guard']['roles']['cols'] = const(8192)
    _, python, _ = emit_runtime_guards(c, {}, sig, 32)
    assert '_arg4, 8192, _arg2' in python
    with pytest.raises(RuntimeError, match='signature'):
        emit_runtime_guards(c, {}, {**sig, 4:'i64'}, 32)
    c['guard']['roles']['rows'] = arg(99)
    with pytest.raises(RuntimeError, match='signature'):
        emit_runtime_guards(c, {}, sig, 32)


@pytest.mark.parametrize('limit', [0, -1, True, None, 1 << 31])
def test_invalid_physical_limit(limit):
    with pytest.raises(RuntimeError):
        emit_runtime_guards(CONTRACT, {}, {0:'*fp16',1:'*fp16',2:'i32',3:'i32'}, limit)


@pytest.mark.parametrize('fixture', [CONTRACT, JAGGED_CONTRACT, FLAT_CONTRACT, FIXED_CONTRACT])
@pytest.mark.parametrize('auto_map', [False, True])
def test_generated_driver_preserves_logical_grid_and_physical_mapping(fixture, auto_map):
    path = Path(__file__).resolve().parents[2] / 'backend' / 'driver.py'
    spec = importlib.util.spec_from_file_location('runtime_split_driver', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    roles = fixture['guard']['roles']
    signature = {r['arg']-2:r['type'] for r in roles['inputs'] +
                 [r for k,r in roles.items() if k != 'inputs' and 'arg' in r]}
    signature = dict(sorted(signature.items()))
    meta = SimpleNamespace(target=driver_target(), workspace_size=0, lock_init_value=0, lock_num=0,
        bs_task_type=0, mix_mode='aiv', shared=0, compile_on_910_95=True,
        parallel_mode='simd', is_pure_simt=False, debug=False,
        program_grid_mapping_applied=False, program_grid_transforms=None,
        row_coalescing_applied=False, auto_blockify_enabled=False, ptsm_cap_authorized=False,
        coalesce_factor=1, coalesce_axis=-1, l1_program_split_kind='split_mix_half_grid',
        l1_program_split_schema=SCHEMA_VERSION, l1_program_split_runtime_contract=fixture)
    meta.auto_blockify_enabled = auto_map
    with patch.object(driver, 'NPUUtils') as npu, \
         patch.object(driver, 'is_ffts_supported', return_value=False), \
         patch.object(driver, 'force_disable_ffts', return_value=False), \
         patch.object(driver, 'get_backend_func', side_effect=lambda *a:'/* mocked */'):
        npu.return_value.get_aivector_core_num.return_value = 64
        npu.return_value.get_aicore_num.return_value = 32
        source = driver.make_launcher({}, signature, meta)
        meta.coalesce_factor = 2
        meta.coalesce_axis = 0
        meta.row_coalescing_applied = True
        with pytest.raises(RuntimeError, match='cannot coalesce'):
            driver.make_launcher({}, signature, meta)
    from triton.backends.ascend.l1_launch_contract import physical_block_count
    assert source.count('uint32_t blockNum = '+physical_block_count(fixture)+';') == 2
    assert source.count('if (!valid_') >= 2
    # Python guards reference parsed _arg values and return a PyObject* error.
    # They must not be emitted into Maple4's void, scalar-argument helper.
    helper = source.split('static void _launch(', 1)[1].split('static PyObject* launch(', 1)[0]
    python_entry = source.split('static PyObject* launch(', 1)[1]
    assert 'if (!valid_' not in helper
    assert 'if (!valid_' in python_entry
    assert python_entry.index('if (!valid_') < python_entry.index('_launch(kernelName')
    assert source.index('num_args !=') < source.index('copied_kernel_args.emplace_back')


def test_fixed_grid_and_ordinary_guards(tmp_path):
    definition, _, _ = emit_runtime_guards(FIXED_CONTRACT, {}, {0:'*fp32',1:'*fp32'},32)
    src = tmp_path/'fixed.cpp'
    src.write_text(definition+'\nextern "C" bool check(int n,int y,int z) { return valid_fixed_program_split(n,y,z); }')
    so = tmp_path/'fixed.so'
    subprocess.run(['c++','-shared','-fPIC',str(src),'-o',str(so)],check=True)
    check = ctypes.CDLL(str(so)).check
    check.argtypes, check.restype = [ctypes.c_int]*3, ctypes.c_bool
    assert check(4,1,1)
    assert not check(2,1,1)
    assert not check(4,2,1)
    assert emit_launch_guards(SimpleNamespace(), {}, {},32) == ('','','')


@pytest.mark.parametrize("specialized", [False, True])
@pytest.mark.parametrize("tile_mode", ["full", "partial"])
def test_three_input_reordered_constant_rows_launchers(tmp_path, specialized, tile_mode):
    """Layout emitted by the backend's generalized three-input fixture.

    Rows is constant, cols/stride share one scalar, output precedes the inputs,
    and two unrelated scalars plus a pointer remain in the compiled signature.
    """
    import sysconfig
    contract = {
        'launch': {'distribution': 'half_grid', 'allow_odd_grid': False},
        'guard': {'kind': 'tiles_2d', 'tile_mode': tile_mode,
                  'tile': {'rows': 16, 'cols': 1344, 'element_bits': 16},
                  'roles': {'inputs': [arg(3, '*fp16'), arg(4, '*fp16'), arg(5, '*fp16')],
                            'output': arg(2, '*fp16'), 'rows': const(80),
                            'cols': arg(6), 'stride': arg(6)}}}
    names = ['out', 'y', 'z', 'x', 'stride', 'unused_rows', 'unused_cols', 'unused_ptr']
    types = ['*fp16'] * 4 + ['i32'] * 3 + ['*fp16']
    constants = {}
    if specialized:
        import triton
        from triton._C.libtriton import ir
        from triton.compiler.compiler import ASTSource
        from triton.backends.ascend.compiler import NPUOptions

        @triton.jit
        def specialized_inputs(out, y, unit, z, x, stride, unused_rows, unused_cols, unused_ptr):
            value = (tl.load(x) + tl.load(y)).to(tl.float16) + tl.load(z)
            tl.store(out + stride, value + (unit - 1))

        names.insert(2, 'unit')
        types.insert(2, 'constexpr')
        constants = {2: 1}
        ast_source = ASTSource(specialized_inputs, dict(zip(names, types)), constexprs={'unit': 1})
        context = ir.context()
        ir.load_dialects(context)
        module = ast_source.make_ir(driver_target(), NPUOptions(), {}, {}, context)
        function = module.get_function('specialized_inputs')
        assert function.args(7) is not None
        with pytest.raises(IndexError):
            function.args(8)
        assert 'tt.addptr %out, %stride' in str(module)
    signature = dict(enumerate(types))
    stride_name = f'_arg{names.index("stride")}'
    definition, python_guard, c_guard = emit_runtime_guards(contract, constants, signature, 32)
    assert f'80, {stride_name}, {stride_name}' in python_guard
    assert 'memcpy(&split_cols, kernel_args[4]' in c_guard
    assert 'memcpy(&split_stride, kernel_args[4]' in c_guard
    assert 'int32_t split_rows = 80;' in c_guard
    assert 'num_args != 8' in c_guard
    assert all(f'kernel_args[{i}]' not in c_guard for i in (0, 1, 2, 3, 5, 6, 7))

    source = tmp_path / 'three_input_guards.cpp'
    source.write_text('#include <Python.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n' +
        definition + '\nextern "C" PyObject* python_check(int gridX,int gridY,int gridZ,int32_t ' + stride_name + ') {\n' +
        python_guard + '\nPy_RETURN_TRUE;\n}\n' +
        'extern "C" int c_check(int gridX,int gridY,int gridZ,int num_args,'
        'const void **kernel_args,const size_t *arg_sizes) {\nint dispatched=0;\n'
        'auto call=[&]() {\n' + c_guard + '\ndispatched=1;\n};call();return dispatched;\n}\n')
    shared = tmp_path / 'three_input_guards.so'
    subprocess.run(['c++', '-shared', '-fPIC', '-I' + sysconfig.get_path('include'),
                    str(source), '-o', str(shared)], check=True)
    library = ctypes.PyDLL(str(shared))
    python = library.python_check
    python.argtypes, python.restype = [ctypes.c_int] * 4, ctypes.py_object
    c_api = library.c_check
    c_api.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    c_api.restype = ctypes.c_int
    stride = ctypes.c_int32(2688)
    # The unrelated scalar slots contain invalid dimensions. They must be ignored.
    unused_rows, unused_cols = ctypes.c_int32(-1), ctypes.c_int32(1)
    pointers = (ctypes.c_void_p * 8)(1, 2, 3, 4, ctypes.addressof(stride),
                                    ctypes.addressof(unused_rows), ctypes.addressof(unused_cols), 5)
    sizes = (ctypes.c_size_t * 8)(8, 8, 8, 8, 4, 4, 4, 8)
    assert python(4, 1, 1, stride.value) is True
    assert c_api(4, 1, 1, 8, pointers, sizes) == 1
    for value in (1, 7, 2689):
        stride.value = value
        if tile_mode == 'partial':
            assert python(4, 1, 1, value) is True
            assert c_api(4, 1, 1, 8, pointers, sizes) == 1
        else:
            with pytest.raises(ValueError):
                python(4, 1, 1, value)
            assert c_api(4, 1, 1, 8, pointers, sizes) == 0
    for invalid in (0, -1, (1 << 31) - 1):
        stride.value = invalid
        with pytest.raises(ValueError):
            python(4, 1, 1, invalid)
        assert c_api(4, 1, 1, 8, pointers, sizes) == 0
    stride.value = 2688
    sizes[4] = 8
    assert c_api(4, 1, 1, 8, pointers, sizes) == 0

    path = Path(__file__).resolve().parents[2] / 'backend' / 'driver.py'
    spec = importlib.util.spec_from_file_location('three_input_driver', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    meta = SimpleNamespace(target=driver_target(), workspace_size=0, lock_init_value=0, lock_num=0,
        bs_task_type=0, mix_mode='aiv', shared=0, compile_on_910_95=True,
        parallel_mode='simd', is_pure_simt=False, debug=False,
        program_grid_mapping_applied=False, program_grid_transforms=None,
        row_coalescing_applied=False, auto_blockify_enabled=False, ptsm_cap_authorized=False,
        coalesce_factor=1, coalesce_axis=-1, l1_program_split_kind='split_mix_half_grid',
        l1_program_split_schema=SCHEMA_VERSION, l1_program_split_runtime_contract=contract)
    launcher = driver.NPULauncher.__new__(driver.NPULauncher)
    launcher.src = SimpleNamespace(fn=SimpleNamespace(arg_names=names),
                                  constants={(i,): value for i, value in constants.items()},
                                  signature={names[i]: value for i, value in signature.items()})
    launcher.metadata = meta
    with patch.object(driver, 'NPUUtils') as npu, \
         patch.object(driver, 'is_ffts_supported', return_value=False), \
         patch.object(driver, 'force_disable_ffts', return_value=False), \
         patch.object(driver, 'get_backend_func', side_effect=lambda *a: '/* mocked */'), \
         patch.object(driver, 'make_npu_launcher_stub', return_value='unused.so') as build:
        npu.return_value.get_aivector_core_num.return_value = 64
        npu.return_value.get_aicore_num.return_value = 32
        assert launcher._make_launcher_stub_path() == 'unused.so'
    wrapper = build.call_args.args[1]
    assert python_guard in wrapper and c_guard in wrapper
    assert 'void triton_launch_kernel(' in wrapper
    assert wrapper.count('uint32_t blockNum = gridX / 2;') == 2
