llama_add_compile_flags()

add_executable(qwen-pp-trace
    qwen_pp_trace.cpp
)

target_link_libraries(qwen-pp-trace PRIVATE llama-common Threads::Threads)

# Backends are loaded dynamically at runtime via ggml_backend_load_all()

set_target_properties(qwen-pp-trace PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)