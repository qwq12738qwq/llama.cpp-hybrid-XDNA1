cmake_minimum_required(VERSION 3.20)
project(simple-xdna-test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Find llama.cpp libraries
set(LLAMA_CPP_BUILD_DIR "/workspace/llama.cpp/build-new" CACHE PATH "llama.cpp build directory")

# Include directories
include_directories(${LLAMA_CPP_BUILD_DIR}/ggml/include)
include_directories(${LLAMA_CPP_BUILD_DIR}/include)
include_directories(/workspace/llama.cpp/ggml/include)
include_directories(/workspace/llama.cpp/include)

# Source
add_executable(simple-xdna-test simple_xdna_test.cpp)

# Link libraries
target_link_libraries(simple-xdna-test PRIVATE
    ${LLAMA_CPP_BUILD_DIR}/bin/libggml.so
    ${LLAMA_CPP_BUILD_DIR}/bin/libggml-base.so
    ${LLAMA_CPP_BUILD_DIR}/bin/libggml-cpu.so
    ${LLAMA_CPP_BUILD_DIR}/bin/libllama-common.so
    Threads::Threads
    dl
    rt
    pthread
)

# Suppress specific warnings treated as errors for this test
target_compile_options(simple-xdna-test PRIVATE
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

set_target_properties(simple-xdna-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LLAMA_CPP_BUILD_DIR}/bin
)