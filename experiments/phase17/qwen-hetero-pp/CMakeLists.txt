llama_add_compile_flags()

add_executable(qwen-hetero-pp
    qwen_heterogeneous_pp.cpp
)

target_link_libraries(qwen-hetero-pp PRIVATE llama-common Threads::Threads)

# Suppress specific warnings treated as errors for this test
target_compile_options(qwen-hetero-pp PRIVATE 
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

set_target_properties(qwen-hetero-pp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)