



python_add_library(RoughPyComputeModule MODULE WITH_SOABI)

set_target_properties(RoughPyComputeModule PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        LIBRARY_OUTPUT_NAME _rpy_compute_internals
)

target_include_directories(RoughPyComputeModule PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../..
)

target_link_libraries(RoughPyComputeModule
        PRIVATE
        RoughPy::PyCore
)



target_sources(RoughPyComputeModule PRIVATE
        _src/call_config.hpp
        _src/call_config.cpp
        _src/check_dims.hpp
        _src/dense_basic.cpp
        _src/dense_basic.h
        _src/dense_intermediate.cpp
        _src/dense_intermediate.h
        _src/lie_basis.c
        _src/lie_basis.h
        _src/lie_multiplication_cache.cpp
        _src/lie_multiplication_cache.h
        _src/py_binary_array_fn.hpp
        _src/py_ternary_array_fn.hpp
        _src/roughpy_compute_module.c
        _src/sparse_matrix.c
        _src/sparse_matrix.h
        _src/tensor_basis.c
        _src/tensor_basis.h
)



