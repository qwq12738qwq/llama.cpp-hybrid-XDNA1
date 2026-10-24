llama_add_compile_flags()

add_executable(xdna-phase17b-harness
    xdna_phase17b_harness.cpp
)

target_link_libraries(xdna-phase17b-harness PRIVATE ggml ggml-xdna ggml-base ggml-cpu)

target_compile_options(xdna-phase17b-harness PRIVATE
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

set_target_properties(xdna-phase17b-harness PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
