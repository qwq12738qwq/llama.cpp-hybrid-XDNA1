llama_add_compile_flags()

add_executable(xdna-raw-int8-oracle
    xdna_raw_int8_oracle.cpp
)

target_link_libraries(xdna-raw-int8-oracle PRIVATE ggml ggml-xdna ggml-base ggml-cpu)

target_compile_options(xdna-raw-int8-oracle PRIVATE
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

set_target_properties(xdna-raw-int8-oracle PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
