llama_add_compile_flags()

add_executable(xdna-q8_0-q8_0-test
    xdna_q8_0_q8_0_test.cpp
)

target_link_libraries(xdna-q8_0-q8_0-test PRIVATE llama-common Threads::Threads)

target_compile_options(xdna-q8_0-q8_0-test PRIVATE
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

target_link_libraries(xdna-q8_0-q8_0-test PRIVATE ggml ggml-xdna ggml-base ggml-cpu)

set_target_properties(xdna-q8_0-q8_0-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)