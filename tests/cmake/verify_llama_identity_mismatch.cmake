include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/ZooKeeperLlamaIdentity.cmake")

set(LLAMA_BUILD_COMMIT "actual-llama")
set(LLAMA_BUILD_NUMBER "8992")
zoo_verify_llama_build_identity("expected-llama" "8992")
