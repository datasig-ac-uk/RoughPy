



python_add_library(RoughPyComputeModule MODULE WITH_SOABI)

set_target_properties(RoughPyComputeModule PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        LIBRARY_OUTPUT_NAME _rpy_compute_internals
)

target_include_directories(RoughPyComputeModule PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../..
)

target_link_libraries(RoughPyComputeModule PRIVATE Python::Python)


target_sources(RoughPyComputeModule PRIVATE
        _src/dense_basic.cpp
)