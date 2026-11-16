add_triton_library(L1SoftwarePrefetchCommon
  L1PrefetchAttachNoalias.cpp

  DEPENDS
  L1SoftwarePrefetchPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
