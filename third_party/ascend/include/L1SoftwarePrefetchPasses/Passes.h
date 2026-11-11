/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef TRITON_ADAPTER_L1_SOFTWARE_PREFETCH_PASSES_H
#define TRITON_ADAPTER_L1_SOFTWARE_PREFETCH_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

#define GEN_PASS_DECL_ATTACHPOINTERARGSNOALIAS
#include "ascend/include/L1SoftwarePrefetchPasses/Passes.h.inc"

#define GEN_PASS_DEF_ATTACHPOINTERARGSNOALIAS
#include "ascend/include/L1SoftwarePrefetchPasses/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAttachPointerArgsNoAliasPass(
    const AttachPointerArgsNoAliasOptions &options = {});

#define GEN_PASS_REGISTRATION
#include "ascend/include/L1SoftwarePrefetchPasses/Passes.h.inc"

} // namespace triton
} // namespace mlir

namespace {

class AttachPointerArgsNoAliasPass
    : public ::impl::AttachPointerArgsNoAliasBase<AttachPointerArgsNoAliasPass> {
public:
  explicit AttachPointerArgsNoAliasPass(const AttachPointerArgsNoAliasOptions &options);
  void runOnOperation() override;
};

} // namespace

#endif // TRITON_ADAPTER_L1_SOFTWARE_PREFETCH_PASSES_H
