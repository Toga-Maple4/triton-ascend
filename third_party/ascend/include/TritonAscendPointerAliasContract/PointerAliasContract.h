#ifndef TRITON_ASCEND_POINTER_ALIAS_CONTRACT_H
#define TRITON_ASCEND_POINTER_ALIAS_CONTRACT_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::ascend {

/// IR contract representing a host/ABI noalias promise between function pointer
/// arguments. The attachment pass materializes the caller's promise for
/// New-L1; the backend consumes the preserved attribute.
inline constexpr llvm::StringRef kPointerArgsNoAliasAttr = "ascend.memory.pointer_args_noalias";

} // namespace mlir::triton::ascend

#endif // TRITON_ASCEND_POINTER_ALIAS_CONTRACT_H
