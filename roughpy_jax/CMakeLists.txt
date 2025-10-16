project(RoughPy_JaxModule LANGUAGES CXX)

# Optionally enable CUDA build
include(CheckLanguage)
check_language(CUDA)

if (ROUGHPY_JAX_CUDA)
    if (NOT DEFINED CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CMAKE_CUDA_COMPILER must be available for ROUGHPY_JAX_CUDA")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 90 120)
    endif()

    find_package(CUDAToolkit REQUIRED)
endif()

# Get XLA includes via jax lib
# FIXME before merge to to main: remove find_package duplicated here from
# roughpy/ dir by moving python find into into cmake/ dir function, and include
# find_python in top level cmake.
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}"
            "-c" "from jax import ffi; print(ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR
)
message(STATUS "XLA include directory: ${XLA_DIR}")

add_library(RoughPy_JaxModule SHARED
    src/cpu/dense_ft_fma.cpp
    src/cpu/dense_ft_exp.cpp
)

# Ensure that XLA FFI definitions are visible in genreated .so
set_target_properties(RoughPy_JaxModule PROPERTIES
    CXX_VISIBILITY_PRESET default
)

# FIXME for review, copied from RoughPy_PyModule which is missing platforms
if (WIN32)
elseif (APPLE)
else ()
    set_target_properties(RoughPy_JaxModule PROPERTIES
        INSTALL_RPATH $ORIGIN
    )
endif ()

target_include_directories(RoughPy_JaxModule PRIVATE
    ${XLA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/..
)

# Use same naming convention as rpy_compute (except without cpython versioning)
set_target_properties(RoughPy_JaxModule PROPERTIES PREFIX "")
set_target_properties(RoughPy_JaxModule PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    LIBRARY_OUTPUT_NAME _rpy_jax_internals
)
