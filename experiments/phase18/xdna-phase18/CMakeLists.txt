cmake_minimum_required(VERSION 3.16)
project(xdna-phase18 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# Find XRT
find_path(XRT_INCLUDE_DIR xrt/xrt_device.h PATHS /usr/include /opt/xilinx/xrt/include)
find_library(XRT_LIBRARY NAMES xrt_coreutil PATHS /usr/lib/x86_64-linux-gnu /opt/xilinx/xrt/lib)

# GGML paths - inside the llama.cpp source tree
set(GGML_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include")
set(GGML_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src")
set(GGML_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build-phase17b/bin")
set(GGML_LIB_DIR2 "${CMAKE_CURRENT_SOURCE_DIR}/../../build/bin")

add_executable(xdna-phase18-harness xdna_phase18_harness.cpp)

target_include_directories(xdna-phase18-harness PRIVATE
    ${GGML_INCLUDE}
    ${GGML_SRC}
    ${XRT_INCLUDE_DIR}
)

find_library(GGML_LIB ggml HINTS ${GGML_LIB_DIR} ${GGML_LIB_DIR2})
find_library(GGML_BASE_LIB ggml-base HINTS ${GGML_LIB_DIR} ${GGML_LIB_DIR2})
find_library(GGML_CPU_LIB ggml-cpu HINTS ${GGML_LIB_DIR} ${GGML_LIB_DIR2})
find_library(GGML_VK_LIB ggml-vulkan HINTS ${GGML_LIB_DIR} ${GGML_LIB_DIR2})
find_library(GGML_XDNA_LIB ggml-xdna HINTS ${GGML_LIB_DIR} ${GGML_LIB_DIR2})

target_link_libraries(xdna-phase18-harness PRIVATE
    ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB}
    ${GGML_VK_LIB} ${GGML_XDNA_LIB} ${XRT_LIBRARY}
    pthread dl
)

install(TARGETS xdna-phase18-harness RUNTIME DESTINATION bin)
