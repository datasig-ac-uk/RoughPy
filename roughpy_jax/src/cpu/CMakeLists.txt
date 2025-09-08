nanobind_add_module(_roughpy_jax_cpu NB_STATIC 
    dense_ft_fma.cpp
)

target_include_directories(_roughpy_jax_cpu PRIVATE
    ${XLA_DIR}
)

set_target_properties(_roughpy_jax_cpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    LIBRARY_OUTPUT_NAME _roughpy_jax_cpu
)
