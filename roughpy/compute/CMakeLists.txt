



python_add_library(RoughPyComputeModule MODULE WITH_SOABI)

set_target_properties(RoughPyComputeModule PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        LIBRARY_OUTPUT_NAME _rpy_compute_internals
)

target_include_directories(RoughPyComputeModule PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../..
)

#target_link_libraries(RoughPyComputeModule PRIVATE )


target_sources(RoughPyComputeModule PRIVATE
        _src/algebra_config.hpp
        _src/check_dims.hpp
        _src/dense_basic.cpp
        _src/dense_basic.h
        _src/dense_intermediate.h
        _src/py_binary_array_fn.hpp
        _src/py_headers.h
        _src/py_ternary_array_fn.hpp
        _src/roughpy_compute_module.cpp
)