"""Schema-2 launch facts and runtime premises for program-split L1.

Access/effect completeness and physical feasibility belong to the backend.
Roles describe only values consumed by these checks; unrelated arguments need
no roles. This module never selects an alternative kernel or changes its work.
"""
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1
SCHEMA_VERSION = 2
SPLIT_KIND = "split_mix_half_grid"
# Runtime predicate identities, independent of internal recognizer names.
REQUIRED_ROLES = {
    "fixed_programs": (),
    "persistent_tiles": ("logical_programs", "persistent_steps", "row_tasks", "stride"),
    "tiles_2d": ("rows", "cols", "stride"),
    "flat_tiles": ("size",),
    "program_tiles": ("input_strides",),
    "metadata_bounded_tiles": ("input_stride", "stride", "matrix_stride", "offsets"),
}
POINTER_BITS = {"*fp16": 16, "*bf16": 16, "*fp32": 32, "*fp64": 64,
                "*i8": 8, "*i16": 16, "*i32": 32, "*i64": 64}


def _require(condition, message="invalid runtime program-split contract"):
    if not condition:
        raise RuntimeError(message)


def validate_runtime_contract(contract):
    _require(isinstance(contract, dict) and set(contract) == {"launch", "guard"})
    launch, guard = contract["launch"], contract["guard"]
    _require(isinstance(launch, dict) and isinstance(guard, dict))
    _require(set(guard) == {"kind", "tile", "tile_mode", "roles"})
    kind = guard["kind"]
    _require(isinstance(kind, str) and kind in REQUIRED_ROLES)
    _require(guard["tile_mode"] in ("full", "partial"))
    tile = guard["tile"]
    _require(isinstance(tile, dict) and set(tile) == {"rows", "cols", "element_bits"})
    # Bounds protect the C arithmetic generated below, not DMA/L1 policy.
    _require(all(type(tile[k]) is int and 0 < tile[k] <= INT32_MAX
                 for k in ("rows", "cols", "element_bits")))
    _require(tile["element_bits"] in POINTER_BITS.values())
    mode = launch.get("distribution")
    extra = {"fixed": {"logical_programs"}, "half_grid": set(),
             "resident_pairs": {"resident_stride"}}
    _require(isinstance(mode, str) and mode in extra)
    grid_fields = {"logical_grid"} if kind == "program_tiles" else set()
    _require(set(launch) == {"distribution", "allow_odd_grid"} | extra[mode] | grid_fields)
    if grid_fields:
        _require(launch["logical_grid"] == "xyz")
    _require(type(launch["allow_odd_grid"]) is bool)
    if mode == "fixed":
        n = launch["logical_programs"]
        _require(type(n) is int and 0 < n <= INT32_MAX and n % 2 == 0)
    if mode == "resident_pairs":
        n = launch["resident_stride"]
        _require(type(n) is int and 0 < n <= INT32_MAX // 2)
    # Stage-c compatibility boundary: preserve the existing physical schedules.
    # This coupling is not intrinsic to schema 2; schedule unification must lift
    # it together with the backend realization and its runtime premises.
    expected = "fixed" if kind == "fixed_programs" else "half_grid" if kind in ("tiles_2d", "persistent_tiles") else "resident_pairs"
    _require(mode == expected and launch["allow_odd_grid"] == (kind in ("flat_tiles", "program_tiles")))
    if kind == "persistent_tiles":
        _require(guard["tile_mode"] == "full")
    if kind == "flat_tiles":
        _require(tile["rows"] == 1 and guard["tile_mode"] == "partial")
    if kind == "metadata_bounded_tiles":
        _require(guard["tile_mode"] == "partial")
    roles = guard["roles"]
    _require(isinstance(roles, dict) and set(roles) == {"inputs", "output"} | set(REQUIRED_ROLES[kind]))
    _require(isinstance(roles["inputs"], list) and bool(roles["inputs"]))
    if kind == "program_tiles":
        _require(guard["tile_mode"] == "partial" and
                 isinstance(roles["input_strides"], list) and
                 len(roles["input_strides"]) == len(roles["inputs"]))
    named = _named_roles(roles)
    pointers = []
    for name, role in named:
        _require(isinstance(role, dict))
        # Exactly one source; never give arg/const precedence.
        _require(set(role) in ({"arg", "type"}, {"const", "type"}) or
                 kind == "persistent_tiles" and set(role) == {"grid", "type"})
        pointer = name in ("input", "output", "offsets")
        if pointer:
            _require("arg" in role and isinstance(role["type"], str) and role["type"] in POINTER_BITS)
            pointers.append(role["arg"])
        else:
            _require(role["type"] == "i32")
        if "arg" in role:
            _require(type(role["arg"]) is int and 2 <= role["arg"] <= INT32_MAX)
        elif "grid" in role:
            _require(not pointer and role["grid"] in ("x", "y", "z"))
        else:
            _require(type(role["const"]) is int and -(1 << 31) <= role["const"] <= INT32_MAX)
    if kind == "persistent_tiles":
        stride = roles["stride"]
        _require(set(stride) == {"const", "type"} and stride["const"] > 0
                 and stride["const"] % tile["cols"] == 0)
    _require(len(set(pointers)) == len(pointers), "overlapping program-split pointer roles")
    payloads = roles["inputs"] + ([] if kind == "program_tiles" else [roles["output"]])
    _require(len({r["type"] for r in payloads}) == 1 and
             all(POINTER_BITS[r["type"]] == tile["element_bits"] for r in payloads))


def _named_roles(roles):
    return ([("input", r) for r in roles["inputs"]] +
            [(f"input_stride_{i}", r) for i, r in enumerate(roles.get("input_strides", []))] +
            [(k, v) for k, v in roles.items() if k not in ("inputs", "input_strides")])


def validate_profitability(facts):
    """Validate reporting facts; the backend alone decides whether to retry.

    This module participates in triton_key, so changing this policy invalidates
    previously cached kernels, including binaries compiled without the filter.
    """
    import math
    _require(isinstance(facts, dict) and set(facts) == {
        "version", "threshold", "decision", "reason", "inputs", "compute_score"},
        "missing or invalid New-L1 profitability facts; rebuild compiler")
    score = facts["compute_score"]
    _require(type(facts["version"]) is int and facts["version"] == 2
             and type(facts["threshold"]) is int and facts["threshold"] == 40
             and type(facts["inputs"]) is int and facts["inputs"] >= 0
             and isinstance(facts["reason"], str)
             and (score is None or type(score) in (int, float)
                  and math.isfinite(score) and score >= 0),
             "invalid New-L1 profitability facts")
    decision = "unknown" if score is None else "filtered" if score < 40 else "passed"
    _require(facts["decision"] == decision, "inconsistent New-L1 profitability decision")


def validate_compile_result(result, metadata):
    _require(isinstance(result, dict))
    if type(result.get("schema_version")) is not int or result["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("unsupported program-split result schema; recompile kernel")
    if "profitability" in result:
        validate_profitability(result["profitability"])
        _require(result.get("kind") != SPLIT_KIND or result["profitability"]["decision"] != "filtered",
                 "filtered New-L1 candidate produced a split binary")
        result = {k: v for k, v in result.items() if k != "profitability"}
    if "compile_options" in result:
        options = result["compile_options"]
        flags = {"enable_vf_fusion", "disable_vf_reachable_check", "enable_auto_multi_buffer"}
        _require(isinstance(options, dict) and set(options) == flags | {"hfusion_max_buffer_count_tuning"}
                 and all(type(options[k]) is bool for k in flags)
                 and type(options["hfusion_max_buffer_count_tuning"]) is int,
                 "invalid program-split compile options")
        result = {k: v for k, v in result.items() if k != "compile_options"}
    if result.get("kind") == "ordinary":
        _require(set(result) == {"schema_version", "kind", "reason"} and isinstance(result["reason"], str))
        return
    _require(set(result) == {"schema_version", "kind", "target", "launch", "guard"})
    _require(result["kind"] == SPLIT_KIND and result["target"] == metadata["target"].arch
             and metadata["mix_mode"] == "aiv" and metadata.get("coalesce_factor", 1) == 1,
             "program-split result does not match the compiled kernel")
    validate_runtime_contract({k: result[k] for k in ("launch", "guard")})


def validate_launch_metadata(metadata, coalesce_factor):
    kind = getattr(metadata, "l1_program_split_kind", "ordinary")
    _require(kind in ("ordinary", SPLIT_KIND), "invalid program-split launch kind")
    schema = getattr(metadata, "l1_program_split_schema", None)
    if schema is not None or kind == SPLIT_KIND:
        _require(type(schema) is int and schema == SCHEMA_VERSION,
                 "incompatible cached program-split metadata; recompile kernel")
    contract = getattr(metadata, "l1_program_split_runtime_contract", None)
    if kind == "ordinary":
        _require(contract is None, "runtime contract without accepted MIX binary")
        return
    _require(coalesce_factor == 1, "program-split cannot coalesce logical blocks")
    _require(not any(getattr(metadata, "l1_program_split_" + k, None) is not None for k in
                     ("logical_programs", "physical_programs", "persistent_steps", "row_tasks")),
             "conflicting program-split launch metadata")
    validate_runtime_contract(contract)


def physical_block_count(contract):
    """Interpret only launch mapping, without consulting the guard kind."""
    launch = contract["launch"]
    if launch["distribution"] == "fixed":
        return str(launch["logical_programs"] // 2)
    grid = "((int64_t)gridX * gridY * gridZ)" if launch.get("logical_grid") == "xyz" else "gridX"
    half = f"((int64_t){grid} + 1) / 2" if launch["allow_odd_grid"] else "gridX / 2"
    if launch["distribution"] == "resident_pairs":
        return f"std::min<int64_t>({half}, {launch['resident_stride']})"
    return half


def _bind_roles(contract, constants, signature):
    # Backend positions refer to the specialized ABI, not source parameters.
    # ast_to_ttir removes specialization constants before assigning arguments;
    # the driver's exported-C kernel_args array uses this same compact order.
    # Maple4 ASTSource uses tuple paths for constexpr keys; the flattened
    # launcher signature explicitly labels those arguments as constexpr.
    args = [(i, ty) for i, ty in signature.items()
            if ty != "constexpr" and i not in constants and (i,) not in constants]
    roles = contract["guard"]["roles"]
    values, indices = {}, {}
    for name, role in _named_roles(roles):
        if "const" in role:
            values[name] = str(role["const"])
            continue
        if "grid" in role:
            values[name] = "grid" + role["grid"].upper()
            continue
        index = role["arg"] - 2  # TTAdapter hidden pointers are not public args.
        _require(index < len(args) and args[index][1] == role["type"],
                 "program-split role does not match signature")
        if not role["type"].startswith("*"):
            indices[name] = index
            values[name] = f"_arg{args[index][0]}"
    return args, values, indices


def _emit_checks(predicate, function, names, args, values, indices, message):
    call = ', '.join(["gridX", "gridY", "gridZ"] + [values[n] for n in names])
    python = f'if (!{function}({call})) {{\n    PyErr_SetString(PyExc_ValueError, "{message}");\n    return NULL;\n  }}'
    checks = ''.join(f" || !kernel_args[{i}] || arg_sizes[{i}] != sizeof(int32_t)" for i in sorted(set(indices.values())))
    init = []
    for name in names:
        if name in indices:
            init.append(f"int32_t split_{name} = 0; memcpy(&split_{name}, kernel_args[{indices[name]}], sizeof(split_{name}));")
        else:
            init.append(f"int32_t split_{name} = {values[name]};")
    call = ', '.join(["gridX", "gridY", "gridZ"] + ["split_" + n for n in names])
    c_api = f'if (num_args != {len(args)} || !kernel_args || !arg_sizes{checks}) {{\n    fprintf(stderr, "Error: invalid split MIX arguments\\n");\n    return;\n  }}\n  ' + '\n  '.join(init)
    c_api += f'\n  if (!{function}({call})) {{\n    fprintf(stderr, "Error: {message}\\n");\n    return;\n  }}'
    return predicate, python, c_api


def emit_launch_guards(metadata, constants, signature, max_physical_blocks):
    validate_launch_metadata(metadata, getattr(metadata, "coalesce_factor", 1))
    if getattr(metadata, "l1_program_split_kind", "ordinary") != SPLIT_KIND:
        return "", "", ""
    return emit_runtime_guards(metadata.l1_program_split_runtime_contract, constants, signature, max_physical_blocks)


def emit_runtime_guards(contract, constants, signature, max_physical_blocks):
    validate_runtime_contract(contract)
    _require(type(max_physical_blocks) is int and 0 < max_physical_blocks <= INT32_MAX, "invalid physical MIX core limit")
    args, values, indices = _bind_roles(contract, constants, signature)
    guard, launch = contract["guard"], contract["launch"]
    kind = guard["kind"]
    bm, bn, bits = (guard["tile"][k] for k in ("rows", "cols", "element_bits"))
    if kind == "program_tiles":
        stride = launch["resident_stride"]
        names = tuple(f"input_stride_{i}" for i in range(len(guard["roles"]["inputs"])))
        params = ''.join(f", int64_t {name}" for name in names)
        checks = ' || '.join(f"{name} < {bn} || {name} > {INT32_MAX}LL" for name in names)
        predicate = f'''
static bool valid_program_tiles(int gridX, int gridY, int gridZ{params}) {{
  if (gridX <= 0 || gridY <= 0 || gridZ <= 0) return false;
  // Bound each product before multiplying; backend program IDs are i32.
  int64_t programs = gridX;
  if (programs > {INT32_MAX}LL / gridY) return false;
  programs *= gridY;
  if (programs > {INT32_MAX - 2 * stride}LL / gridZ) return false;
  programs *= gridZ;
  if ({checks}) return false;
  int64_t pairs = (programs + 1) / 2;
  return (pairs < {stride} ? pairs : {stride}) <= {max_physical_blocks};
}}
'''
        return _emit_checks(predicate, "valid_program_tiles", names, args, values, indices,
                            "split MIX program grid/transfer stride violates the runtime contract")
    if kind == "fixed_programs":
        logical = launch["logical_programs"]
        predicate = f"static bool valid_fixed_program_split(int gridX, int gridY, int gridZ) {{ return gridX == {logical} && gridY == 1 && gridZ == 1 && {logical // 2} <= {max_physical_blocks}; }}\n"
        return _emit_checks(predicate, "valid_fixed_program_split", (), args, values, indices,
                            "split MIX requires its compiled logical grid")
    if kind == "persistent_tiles":
        # These premises prove the existing q + N*floor(i/C) partition in its
        # actual i32/index domain, including inactive-payload comparisons.
        stride = guard["roles"]["stride"]["const"]
        columns = stride // bn
        names = ("logical_programs", "persistent_steps", "row_tasks", "stride")
        predicate = f'''
static bool valid_persistent_tiles(int gridX, int gridY, int gridZ,
                                   int64_t logical_programs, int64_t persistent_steps,
                                   int64_t row_tasks, int64_t stride) {{
  if (logical_programs <= 0 || logical_programs > {INT32_MAX}LL ||
      gridX != logical_programs || gridX % 2 || gridY != 1 || gridZ != 1 ||
      gridX / 2 > {max_physical_blocks} ||
      persistent_steps <= 0 || persistent_steps > {INT32_MAX}LL ||
      row_tasks <= 0 || row_tasks > {INT32_MAX}LL || stride != {stride}) return false;
  int64_t round = (persistent_steps - 1) / {columns};
  if (round > ({INT32_MAX}LL - (logical_programs - 1)) / logical_programs) return false;
  int64_t tmax = logical_programs - 1 + logical_programs * round;
  int64_t last = row_tasks - 1 < tmax ? row_tasks - 1 : tmax;
  if (last > {INT32_MAX // bm}LL) return false;
  int64_t column = persistent_steps - 1 - (last / logical_programs) * {columns};
  if (column >= {columns}) column = {columns - 1};
  int64_t tail = column * {bn}LL + {bn - 1};
  int64_t last_row = last * {bm}LL + {bm - 1};
  return last_row <= ({INT64_MAX // (bits // 8)}LL - tail) / stride;
}}
'''
        predicate, py, c = _emit_checks(
            predicate, "valid_persistent_tiles", names, args, values, indices,
            "persistent_launch_domain_violation")
        def diagnostic(n):
            return (f'(gridX != {n} ? "persistent_grid_count_mismatch" : '
                    '"persistent_launch_domain_violation")')
        py = py.replace('"persistent_launch_domain_violation"', diagnostic(values["logical_programs"]))
        c = c.replace('fprintf(stderr, "Error: persistent_launch_domain_violation\\n");',
                      'fprintf(stderr, "Error: %s\\n", ' + diagnostic("split_logical_programs") + ');')
        return predicate, py, c
    if kind == "tiles_2d":
        partial = guard["tile_mode"] == "partial"
        shape_alignment = "false" if partial else f"rows % {bm} || cols % {bn}"
        row_tasks = f"(rows + {bm - 1}) / {bm}" if partial else f"rows / {bm}"
        column_tasks = f"(cols + {bn - 1}) / {bn}" if partial else f"cols / {bn}"
        predicate = f'''
    static bool valid_runtime_program_split(int gridX, int gridY, int gridZ,
                                            int64_t rows, int64_t cols, int64_t stride) {{
      if (gridX <= 0 || gridX % 2 || gridY != 1 || gridZ != 1 ||
          gridX / 2 > {max_physical_blocks} ||
          rows <= 0 || rows > {INT32_MAX - (bm - 1)}LL ||
          cols <= 0 || cols > {INT32_MAX - bn}LL ||
          stride <= 0 || stride > {INT32_MAX}LL || cols > stride ||
          {shape_alignment}) return false;
      int64_t tasks = {row_tasks};
      if (tasks + gridX - 1 > {INT32_MAX // bm}LL) return false;
      int64_t steps = ((tasks + gridX - 1) / gridX) * ({column_tasks});
      // Guard the largest row/byte offset before any signed multiplication.
      if (rows - 1 > ({INT64_MAX // (bits // 8)}LL - cols) / stride) return false;
      return steps > 0 && steps < {INT32_MAX}LL;
    }}
    '''
        return _emit_checks(predicate, "valid_runtime_program_split", ("rows", "cols", "stride"), args, values, indices,
                            "split MIX shape/grid/stride violates the runtime contract or physical core limit")
    if kind == "metadata_bounded_tiles":
        predicate = f'''
    static bool valid_jagged_program_split(int gridX, int gridY, int gridZ,
                                          int64_t input_stride, int64_t stride,
                                          int64_t matrix_stride) {{
      // The compiler iterates logical sequence pairs inside resident MIX trios.
      // The launcher caps physical blocks without discarding logical work.
      // Bound generated i32 arithmetic by the dense extent, not a tested grid size.
      // For positive rows/stride, tiles_per_sequence <= matrix_stride. Each
      // resident group owns at most gridX/2 pairs, so the extent check below also
      // bounds flattened stream steps and their next-step lookahead. Half-grid
      // rounding by pair_stride-1 is safe because pair_stride <= INT32_MAX/2.
      if (gridX <= 0 || gridX % 2 || gridY != 1 || gridZ != 1 ||
          (gridX / 2 < {launch['resident_stride']} ? gridX / 2 : {launch['resident_stride']}) > {max_physical_blocks} ||
          stride <= 0 || stride > {INT32_MAX - bn}LL || input_stride != stride ||
          matrix_stride <= 0 || matrix_stride % stride ||
          matrix_stride > {INT32_MAX}LL / gridX) return false;
      int64_t rows = matrix_stride / stride;
      if (rows > {INT32_MAX - (bm - 1)}LL) return false;
      int64_t steps = ((rows + {bm - 1}) / {bm}) * ((stride + {bn - 1}) / {bn});
      return steps > 0 && steps < {INT32_MAX}LL;
    }}
    '''
        return _emit_checks(predicate, "valid_jagged_program_split", ("input_stride", "stride", "matrix_stride"), args, values, indices,
                            "split MIX jagged shape/grid/stride violates the runtime contract")
    width, stride = bn, launch["resident_stride"]
    predicate = f'''
static bool valid_flat_program_split(int gridX, int gridY, int gridZ, int64_t size) {{
  if (gridX <= 0 || gridX > {INT32_MAX - 2 * stride}LL || gridY != 1 || gridZ != 1 ||
      size < 0 || size > {INT32_MAX}LL ||
      (int64_t)(gridX - 1) * {width}LL > {INT32_MAX}LL) return false;
  int64_t pairs = ((int64_t)gridX + 1) / 2;
  return (pairs < {stride} ? pairs : {stride}) <= {max_physical_blocks};
}}
'''
    return _emit_checks(predicate, "valid_flat_program_split", ("size",), args, values, indices,
                        "split MIX flat-tile shape/grid violates the runtime contract")
