#ifndef TRITON_L1_SOFTWARE_PREFETCH_PASSES
#define TRITON_L1_SOFTWARE_PREFETCH_PASSES

include "mlir/Pass/PassBase.td"

// Materialize the caller's no-alias promise for New-L1. New-L1 recognition
// and transformation are implemented by AscendNPU-IR.
def L1PrefetchAttachNoalias
    : Pass<"l1-prefetch-attach-noalias", "mlir::ModuleOp"> {
  let summary = "Materialize the caller's kernel-pointer no-alias contract onto kernel entries";
  let constructor = "triton::createL1PrefetchAttachNoaliasPass()";
  let options = [Option<"onlyArgumentCount", "only-argument-count", "unsigned", "0",
                         "Restrict contract attachment to this ABI arity (0 = all)">];
}

#endif // TRITON_L1_SOFTWARE_PREFETCH_PASSES
