"""Typed role fixtures. These are interface tests, not backend acceptance claims."""
from copy import deepcopy


def arg(index, type="i32"):
    return {"arg": index, "type": type}


def const(value):
    return {"const": value, "type": "i32"}


def contract(kind="tiles_2d", inputs=(2,), output=3, pointer="*fp16", rows=32, cols=2048,
             mode="full", **roles):
    launch = {"distribution": "half_grid", "allow_odd_grid": False}
    if kind == "fixed_programs":
        launch.update(distribution="fixed", logical_programs=4)
    if kind in ("flat_tiles", "metadata_bounded_tiles"):
        launch.update(distribution="resident_pairs", resident_stride=32,
                      allow_odd_grid=kind == "flat_tiles")
    return {"launch": launch, "guard": {
        "kind": kind, "tile": {"rows": rows, "cols": cols, "element_bits": int(pointer[-2:])},
        "tile_mode": mode, "roles": {"inputs": [arg(i, pointer) for i in inputs],
                                       "output": arg(output, pointer), **roles}}}


def change(c, *, pointer=None, mode=None, cols=None):
    c = deepcopy(c)
    if pointer:
        for role in c["guard"]["roles"]["inputs"] + [c["guard"]["roles"]["output"]]:
            role["type"] = pointer
        c["guard"]["tile"]["element_bits"] = int(pointer[-2:])
    if mode:
        c["guard"]["tile_mode"] = mode
    if cols:
        c["guard"]["tile"]["cols"] = cols
    return c


CONTRACT = contract(**{"rows": 32, "cols": 2048}, **{})
CONTRACT["guard"]["roles"].update(rows=arg(4), cols=arg(5), stride=const(8192))
STATIC_CONTRACT = deepcopy(CONTRACT)
STATIC_CONTRACT["guard"]["roles"]["cols"] = const(8192)
STRIDED_CONTRACT = contract(pointer="*fp32", rows=16)
STRIDED_CONTRACT["guard"]["roles"].update(rows=arg(5), cols=arg(6), stride=arg(4))
JAGGED_CONTRACT = contract("metadata_bounded_tiles", output=5, pointer="*bf16",
                           rows=256, cols=256, mode="partial", offsets=arg(3,"*i64"),
                           input_stride=arg(4), stride=arg(6), matrix_stride=arg(7))
FLAT_CONTRACT = contract("flat_tiles", inputs=(3,4), output=2, rows=1, cols=32768,
                         mode="partial", size=arg(5))
FLAT_CONTRACT["launch"]["resident_stride"] = 28
TWO_INPUT_CONTRACT = contract(inputs=(2,3), output=4, rows=16)
TWO_INPUT_CONTRACT["guard"]["roles"].update(rows=arg(6), cols=arg(7), stride=arg(5))
FIXED_CONTRACT = contract("fixed_programs", pointer="*fp32", rows=16)
