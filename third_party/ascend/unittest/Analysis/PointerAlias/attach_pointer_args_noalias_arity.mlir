// RUN: triton-opt %s '--l1-attach-pointer-args-noalias=only-argument-count=14' | FileCheck %s

// The two-input caller contract is attached to the final TTAdapter ABI only.
// Helpers and other kernel arities do not acquire a promise from this option.

// CHECK-LABEL: func.func @two_input_entry(
// CHECK-SAME: attributes {ascend.memory.pointer_args_noalias, global_kernel = "local"}
func.func @two_input_entry(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {global_kernel = "local"} {
  return
}

// CHECK-LABEL: func.func @one_input_entry(
// CHECK-SAME: attributes {global_kernel = "local"}
func.func @one_input_entry(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {global_kernel = "local"} {
  return
}

// CHECK-LABEL: func.func @helper(
// CHECK-NOT: ascend.memory.pointer_args_noalias
// CHECK: return
func.func @helper(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
  return
}
