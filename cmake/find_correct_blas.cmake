include_guard()


if (NOT ROUGHPY_DISABLE_BLAS)
    if (RPY_ARCH MATCHES "[xX](86(_64)?|64)|[aA][mM][dD]64" AND NOT
            ROUGHPY_PREFER_ACCELERATE)

        # If we're looking for MKL then we might have installed it via pip.
        # To make sure we can find this, let's use Python's importlib metadata to
        # locate the directory containing MKLConfig.cmake and add the relevant
        # directory to the prefix path.
        execute_process(COMMAND ${Python_EXECUTABLE}
                "${CMAKE_CURRENT_LIST_DIR}/tools/python-get-binary-obj-path.py"
                "--directory" "mkl-devel" "cmake/mkl/MKLConfig.cmake"
                RESULT_VARIABLE _python_mkl_dir_found
                OUTPUT_VARIABLE _python_mkl_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
        )
        if (NOT _python_mkl_dir_found AND EXISTS "${_python_mkl_dir}")
            cmake_path(GET _python_mkl_dir PARENT_PATH _python_mkl_dir)
            message(STATUS "Adding MKL dir from pip-installed mkl: ${_python_mkl_dir}")
            list(APPEND CMAKE_PREFIX_PATH "${_python_mkl_dir}")
            if (NOT MKL_ROOT)
                set(MKL_ROOT "${_python_mkl_dir}")
            endif ()
            set(MKL_DIR "${_python_mkl_dir}")
        endif ()

        # Set the variables that determine the actual MKL library that we need to
        # link. At the moment, we force 32-bit addressing on both 32- and 64-bit
        # platforms, statically linked, and using the Intel OMP library (except on
        # Windows).
        if (RPY_ARCH STREQUAL "x86")
            set(MKL_ARCH ia32)
        else ()
            set(MKL_ARCH intel64)
            set(MKL_INTERFACE lp64)
        endif ()

        set(MKL_LINK static)
        if (WIN32)
            # Currently there is no way to get this working with threading
            # on Windows using this distribution of MKL
            set(MKL_THREADING sequential)
        else ()
            set(MKL_THREADING intel_thread)
        endif ()

        find_package(MKL CONFIG)
        if (TARGET MKL::MKL)
            set(RPY_USE_MKL ON CACHE INTERNAL "BLAS/LAPACK provided by mkl")
            add_library(BLAS::BLAS ALIAS MKL::MKL)
            add_library(LAPACK::LAPACK ALIAS MKL::MKL)

            if (DEFINED MKL_OMP_LIB AND MKL_OMP_LIB)
                #            if (APPLE)
                #                foreach (LANG IN ITEMS C CXX)
                #                    set(OpenMP_${LANG}_LIB_NAMES "${MKL_OMP_LIB}" CACHE STRING "libomp location for OpenMP")
                #                endforeach ()
                #            endif ()
                #            set(OpenMP_${MKL_OMP_LIB}_LIBRARY "${MKL_THREAD_LIB}")
            endif ()
        else ()
            set(RPY_USE_MKL OFF CACHE INTERNAL "BLAS/LAPACK not provided by mkl")
            set(BLA_SIZEOF_INTEGER 4)
            set(BLA_STATIC ON)
        endif ()
    elseif (APPLE)
        set(RPY_USE_ACCELERATE ON CACHE INTERNAL "BLAS/LAPACK provided by Accelerate")
        set(BLA_VENDOR Apple)
        set(BLA_SIZEOF_INTEGER 4)
        set(BLA_STATIC ON)
    endif ()

    if (NOT TARGET BLAS::BLAS)
        find_package(BLAS REQUIRED)
    endif ()
    if (NOT TARGET LAPACK::LAPACK)
        find_package(LAPACK REQUIRED)
    endif ()
endif ()
