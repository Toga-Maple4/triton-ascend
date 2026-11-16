/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "ascend/include/L1SoftwarePrefetchPasses/Passes.h"
#include "ascend/include/TritonAscendPointerAliasContract/PointerAliasContract.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "l1-prefetch-attach-noalias"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isKernelEntry(Operation *operation)
{
    if (auto tritonFunc = dyn_cast<triton::FuncOp>(operation))
        return cast<FunctionOpInterface>(tritonFunc.getOperation()).isPublic();
    if (auto func = dyn_cast<func::FuncOp>(operation))
        return func->hasAttr("global_kernel");
    return false;
}

} // namespace

/// Materializes the caller's promise that the kernel's pointer arguments do not
/// alias. The promise arrives as the `l1_pointer_args_noalias` kernel
/// compilation argument; enabling L1 prefetch implies it.
///
/// This is the ONLY writer of the contract. The analyses read it and never
/// synthesize it, so the contract cannot be conjured by the very pass that
/// benefits from believing it.
L1PrefetchAttachNoaliasPass::L1PrefetchAttachNoaliasPass(
    const L1PrefetchAttachNoaliasOptions &options)
    : L1PrefetchAttachNoaliasBase(options) {}

void L1PrefetchAttachNoaliasPass::runOnOperation()
{
    ModuleOp module = getOperation();
    module.walk([&](FunctionOpInterface function) {
        if (!isKernelEntry(function.getOperation()) || function->hasAttr(ascend::kPointerArgsNoAliasAttr))
            return;
        if (onlyArgumentCount && function.getNumArguments() != onlyArgumentCount)
            return;
        function->setAttr(ascend::kPointerArgsNoAliasAttr, UnitAttr::get(module.getContext()));
        LLVM_DEBUG(llvm::dbgs() << "[l1-prefetch-attach-noalias] " << function.getName() << "\n");
    });
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createL1PrefetchAttachNoaliasPass(
    const L1PrefetchAttachNoaliasOptions &options)
{
    return std::make_unique<L1PrefetchAttachNoaliasPass>(options);
}
