llama_add_compile_flags()

add_executable(xdna-shape-regression
    xdna_shape_regression.cpp
)

target_link_libraries(xdna-shape-regression PRIVATE llama-common Threads::Threads)

target_compile_options(xdna-shape-regression PRIVATE
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

target_link_libraries(xdna-shape-regression PRIVATE ggml ggml-xdna ggml-base ggml-cpu llama-common)

set_target_properties(xdna-shape-regression PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)