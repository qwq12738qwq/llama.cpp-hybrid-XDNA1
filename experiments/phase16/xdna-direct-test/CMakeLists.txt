cmake_minimum_required(VERSION 3.16)
project(xdna-direct-test CXX)

llama_add_compile_flags()

add_executable(xdna-direct-test
    xdna_direct_test.cpp
)

target_link_libraries(xdna-direct-test PRIVATE ggml ggml-xdna ggml-base ggml-cpu)

target_compile_options(xdna-direct-test PRIVATE
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

set_target_properties(xdna-direct-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY 
)
