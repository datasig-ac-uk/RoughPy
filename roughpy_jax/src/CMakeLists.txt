# FIXME for review whether to use nanobind or native python bindings
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(cpu)

if (ROUGHPY_JAX_CUDA)
    add_subdirectory(cuda)
endif()
