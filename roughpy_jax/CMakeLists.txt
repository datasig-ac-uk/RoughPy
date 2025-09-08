project(RoughPy-JAX LANGUAGES CXX)

message(STATUS "Enabling roughpy_jax")

# Optionally enable CUDA build
include(CheckLanguage)
check_language(CUDA)

if (ROUGHPY_JAX_CUDA)
    if (NOT DEFINED CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CMAKE_CUDA_COMPILER must be available for ROUGHPY_JAX_CUDA")
    endif()

    message(STATUS "Enabling CUDA support for roughpy_jax")

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 90 120)
    endif()
endif()

# Get XLA includes via jax lib
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}"
            "-c" "from jax import ffi; print(ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR
)
message(STATUS "XLA include directory: ${XLA_DIR}")

add_subdirectory(src)
