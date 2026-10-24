set(GGML_XDNA_SOURCES
    ggml-xdna.cpp
    xdna-persistent.cpp
    xdna-q8-scale.cpp
    xdna-q8-c4.cpp
    xdna-q8-c4-device-scale.cpp
    xdna-q8-c4-mmul-v2.cpp
    xdna-q8-scale-c4.cpp
    xdna-whole-array.cpp
    xdna-kernel-registry.cpp
    xdna-buffer.cpp
)

set(GGML_XDNA_HEADERS
    ../../include/ggml-xdna.h
    xdna-kernel-registry.h
    xdna-q8-scale.h
    xdna-q8-c4.h
    xdna-q8-c4-device-scale.h
    xdna-q8-c4-mmul-v2.h
    xdna-q8-scale-c4.h
    xdna-whole-array.h
    xdna-buffer.h
)

ggml_add_backend_library(ggml-xdna ${GGML_XDNA_SOURCES} ${GGML_XDNA_HEADERS})

target_include_directories(ggml-xdna PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if (GGML_XDNA)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(XRT REQUIRED xrt)
    target_link_libraries(ggml-xdna PRIVATE ${XRT_LIBRARIES} /usr/lib/x86_64-linux-gnu/libxrt_driver_xdna.so.2)
    target_include_directories(ggml-xdna PRIVATE ${XRT_INCLUDE_DIRS} /usr/include /usr/include/xrt/deprecated ${CMAKE_CURRENT_SOURCE_DIR}/stub)
    target_compile_definitions(ggml-xdna PRIVATE GGML_USE_XDNA GGML_XDNA_PERSISTENT_DEBUG)
endif()
