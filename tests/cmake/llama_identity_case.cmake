include("${ZOO_PROJECT_SOURCE_DIR}/cmake/ZooKeeperLlamaIdentity.cmake")

zoo_verify_llama_build_identity("${EXPECTED_COMMIT}" "${EXPECTED_NUMBER}")
