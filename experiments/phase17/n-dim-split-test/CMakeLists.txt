llama_add_compile_flags()

add_executable(n-dim-split-test
    n_dim_split_test.cpp
)

add_executable(hetero-partition-test
    hetero_partition_test.cpp
)

target_link_libraries(n-dim-split-test PRIVATE llama-common Threads::Threads)
target_link_libraries(hetero-partition-test PRIVATE llama-common Threads::Threads)

# Suppress specific warnings treated as errors for this test
target_compile_options(n-dim-split-test PRIVATE 
    -Wno-error=missing-field-initializers
    -Wno-error=unused-parameter
    -Wno-error=missing-declarations
    -Wno-error=unused-variable
    -Wno-error=unused-but-set-variable
    -Wno-error=missing-braces
    -Wno-missing-field-initializers
    -Wno-missing-declarations
    -Wno-unused-parameter
)

# Backends are loaded dynamically at runtime via ggml_backend_load_all()

set_target_properties(n-dim-split-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)