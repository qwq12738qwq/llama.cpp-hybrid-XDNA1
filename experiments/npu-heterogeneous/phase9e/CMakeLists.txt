cmake_minimum_required(VERSION 3.16)
project(phase9e-scheduler-test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the llama.cpp build
set(LLAMA_CPP_ROOT "/workspace/llama.cpp")
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build")

# Include directories
include_directories(
    ${LLAMA_CPP_ROOT}
    ${LLAMA_CPP_ROOT}/ggml/include
    ${LLAMA_CPP_ROOT}/ggml/src
    ${LLAMA_CPP_ROOT}/common
)

# Source file - use relative path from the build dir
set(TEST_SRC "${LLAMA_CPP_ROOT}/experiments/npu-heterogeneous/phase9e/test_phase9e_scheduler.cpp")

# Create executable
add_executable(test-phase9e-scheduler ${TEST_SRC})

# Link libraries
target_link_libraries(test-phase9e-scheduler PRIVATE
    ${LLAMA_CPP_BUILD}/bin/libllama-common.so.0.4.0
    ${LLAMA_CPP_BUILD}/common/libllama-common-base.a
    ${LLAMA_CPP_BUILD}/bin/libllama.so.0.4.0
    ${LLAMA_CPP_BUILD}/bin/libggml.so.0.23.0
    ${LLAMA_CPP_BUILD}/bin/libggml-cpu.so.0.23.0
    ${LLAMA_CPP_BUILD}/bin/libxdna.so.0.23.0
    ${LLAMA_CPP_BUILD}/bin/libggml-base.so.0.23.0
    pthread
    dl
)

# Set RPATH
set_target_properties(test-phase9e-scheduler PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_BUILD}/bin"
    INSTALL_RPATH "${LLAMA_CPP_BUILD}/bin"
)
