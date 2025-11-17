


add_library(RoughpyFNV1a STATIC
        _src/fnv1a_hash.c
        _src/fnv1a_hash.h
)

set_target_properties(RoughpyFNV1a PROPERTIES
        POSITION_INDEPENDENT_CODE ON)

python_add_library(RoughPyComputeModule MODULE WITH_SOABI)

set_target_properties(RoughPyComputeModule PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        LIBRARY_OUTPUT_NAME _rpy_compute_internals
)

target_include_directories(RoughPyComputeModule PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../..
        ${CMAKE_CURRENT_LIST_DIR}/_src
)

target_link_libraries(RoughPyComputeModule
        PRIVATE
        Python::NumPy
        RoughpyFNV1a
)



target_sources(RoughPyComputeModule PRIVATE
        _src/object_arrays/object_array_iterator.hpp
        _src/object_arrays/object_compute_context.cpp
        _src/object_arrays/object_compute_context.hpp
        _src/call_config.hpp
        _src/call_config.cpp
        _src/call_utils.hpp
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
        _src/py_compat.c
        _src/py_compat.h
        _src/py_fnv1a_hash.h
        _src/py_headers.h
        _src/py_obj_handle.hpp
        _src/py_ternary_array_fn.hpp
        _src/roughpy_compute_module.c
        _src/sparse_matrix.c
        _src/sparse_matrix.h
        _src/tensor_basis.c
        _src/tensor_basis.h
)

