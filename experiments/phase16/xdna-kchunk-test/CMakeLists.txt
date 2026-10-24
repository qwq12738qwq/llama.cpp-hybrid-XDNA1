llama_add_compile_flags()

add_executable(xdna-kchunk-test
    xdna_kchunk_test.cpp
)

target_link_libraries(xdna-kchunk-test PRIVATE ggml ggml-xdna ggml-base ggml-cpu)

target_compile_options(xdna-kchunk-test PRIVATE
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

target_link_libraries(xdna-kchunk-test PRIVATE ggml ggml-xdna ggml-base ggml-cpu)

set_target_properties(xdna-kchunk-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)