add_triton_library(L1SoftwarePrefetchCommon
  AttachPointerArgsNoAliasPass.cpp

  DEPENDS
  L1SoftwarePrefetchPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
