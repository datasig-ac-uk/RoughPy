cmake_minimum_required(VERSION 3.22)

message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)
#list(APPEND VCPKG_INSTALL_OPTIONS "--no-print-usage")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (EXISTS "VERSION.txt")
    file(READ "VERSION.txt" _rpy_version)
    message(STATUS "Repository version ${_rpy_version}")
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _rpy_version "${_rpy_version}")
else ()
    set(_rpy_version 0.0.1)
endif ()

project(RoughPy VERSION ${_rpy_version})

include(CMakeDependentOption)

option(ROUGHPY_BUILD_LA_CONTEXTS "Build the collection of libalgebra contexts" OFF)
option(ROUGHPY_BUILD_TESTS "Build C++ tests for RoughPy" ON)
option(ROUGHPY_BUILD_PYMODULE_INPLACE "Buildg the pymodule in the project roughpy directory" OFF)
option(ROUGHPY_LINK_NUMPY "Link with Numpy library for array handling" ON)
option(ROUGHPY_GENERATE_DEVICE_CODE "Generate code for objects on devices" OFF)
cmake_dependent_option(ROUGHPY_PREFER_ACCELERATE
        "Prefer Accelerate framework on MacOS always" OFF APPLE OFF)

# If testing is enabled, find GTest to make sure the tests can be
# successfully built.
if (ROUGHPY_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)

    if (NOT TARGET GTest::gtest)
        message(FATAL_ERROR "GTest::gtest target not defined")
    endif ()

    enable_testing()

    # If we are building roughpy tests and gtest is available then we also
    # to build the libalgebra_lite tests.
    set(LIBALGEBRA_LITE_BUILD_TESTS ON CACHE INTERNAL "")
endif ()

# If we're building with SKBUILD, we need to define install locations for
# all the components using their special directory variables. Otherwise,
# use the GNUInstall dirs
if (SKBUILD)

    # This is all the variables set by GNUInstallDirs, minus LIBDIR and BINDIR
    set(_ignore_dirs
            SBINDIR
            LIBEXECDIR
            SYSCONFIGDIR
            SHAREDSTATEDIR
            LOCALSTATEDIR
            RUNSTATEDIR
            INCLUDEDIR
            OLDINCLUDEDIR
            DATAROOTDIR
            DATADIR
            INFODIR
            LOCALEDIR
            MANDIR
            DOCDIR
            )

    if (WIN32)
        # On Windows, DLLs are put in BINDIR
        list(APPEND _ignore_dirs LIBDIR)
        set(CMAKE_INSTALL_BINDIR ${SKBUILD_PLATLIB_DIR}/roughpy CACHE STRING
                "Overwritten install for BINDIR")
    else()
        # On not Windows, Shared Objects go in LIBDIR
        list(APPEND _ignore_dirs BINDIR)
        set(CMAKE_INSTALL_LIBDIR ${SKBUILD_PLATLIB_DIR}/roughpy CACHE STRING
                "Overwritten install for LIBDIR")

        list(APPEND _ignore_dirs BINDIR)
    endif()

    foreach(_dir ${_ignore_dirs})
        set(CMAKE_INSTALL_${_dir} ${SKBUILD_NULL_DIR} CACHE STRING
                "Overwritten install for ${_dir}")
    endforeach()

else()
    include(GNUInstallDirs)
endif()


# Load the helper functions
include(cmake/roughpy_helpers.cmake)

# We need to provide some help to make sure we find the correct version of
# Python. Ideally, if we're using Scikit-Build-Core to build the library (via
# pip) and the Python executable is provided via the PYTHON_EXECUTABLE cache
# variable. In this case, make sure that this is the version of Python that gets
# found.
set(PYBIND11_FINDPYTHON ON)
if (NOT PYTHON_FOUND AND SKBUILD)
    cmake_path(GET PYTHON_EXECUTABLE PARENT_PATH _sk_env_dir)
    message(STATUS "SKBuild environment: ${_sk_env_dir}")

    # Some variables that are set might cause issues in the FindPython module,
    # so unset those
    unset(Python_LIBRARY CACHE)
    unset(PYTHON_LIBRARY CACHE)
    unset(Python3_LIBRARY CACHE)

    # clean up temporary
    unset(_sk_env_dir)
else ()
    # If we're not using Scikit-Build-Core (i.e. a pure CMake build) then try
    # looking for a Python virtual environment first.
    set(Python_FIND_VIRTUALENV FIRST)

    # In particular, if ENV{VIRTUAL_ENV} is set then add this to the cmake
    # prefix path so FindPython is more likely to find this environemnt.
    if (DEFINED ENV{VIRTUAL_ENV})
        # Put venv/lib on the prefix path so we can find
        # a pip installed MKL
        message(STATUS "Adding python virtual environment to path")
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}")
    endif ()
endif ()


# At minimum we need Interpreter and Development.Module in order to build a
# Python extension module.
set(PYTHON_COMPONENTS_NEEDED Interpreter Development.Module)
if (ROUGHPY_LINK_NUMPY)
    list(APPEND PYTHON_COMPONENTS_NEEDED NumPy)
endif ()

find_package(Python 3.8 REQUIRED COMPONENTS ${PYTHON_COMPONENTS_NEEDED})



if (NOT DEFINED pybind11_ROOT)
    execute_process(COMMAND
            "${Python_EXECUTABLE}" "-m" "pybind11" "--cmakedir"
            COMMAND_ECHO STDOUT
            RESULT_VARIABLE _python_pybind11_dir_found
            OUTPUT_VARIABLE _python_pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    if (_python_pybind11_dir_found EQUAL 0 AND EXISTS "${_python_pybind11_dir}")
        message(STATUS "Adding Pybind11 cmake dir ${_python_pybind11_dir}")
        set(pybind11_ROOT "${_python_pybind11_dir}" CACHE INTERNAL "")
    endif ()
endif ()

set(RPY_ARCH ${CMAKE_SYSTEM_PROCESSOR})
# TODO: handle this better
if (DEFINED CMAKE_CXX_COMPILER_TARGET)
    set(RPY_ARCH ${CMAKE_CXX_COMPILER_TARGET})
endif ()

set(Boost_NO_WARN_NEW_VERSIONS ON)
find_package(Boost 1.81 REQUIRED COMPONENTS url)
find_package(Eigen3 CONFIG REQUIRED)
find_package(SndFile CONFIG REQUIRED)

# This package is c++17 so cannot be used if changing to a lower standard
find_package(tomlplusplus CONFIG REQUIRED)

message(STATUS "Target architecture ${RPY_ARCH}")
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
        add_library(BLAS::BLAS ALIAS MKL::MKL)
        add_library(LAPACK::LAPACK ALIAS MKL::MKL)

        if (DEFINED MKL_OMP_LIB AND MKL_OMP_LIB)
            if (APPLE)
                foreach (LANG IN ITEMS C CXX)
                    set(OpenMP_${LANG}_FLAGS "-XPreprocessor -fopenmp=${MKL_OMP_LIB}")
                    set(OpenMP_${LANG}_LIB_NAMES "${MKL_OMP_LIB}" CACHE STRING "libomp location for OpenMP")
                endforeach ()
            endif ()
            set(OpenMP_${MKL_OMP_LIB}_LIBRARY "${MKL_THREAD_LIB}")
        endif ()
    else ()
        set(BLA_SIZEOF_INTEGER 4)
        set(BLA_STATIC ON)
    endif ()
elseif (APPLE)
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

if (APPLE AND NOT DEFINED MKL_OMP_LIB)
    get_brew_prefix(_libomp_prefix libomp VERBOSE)

    if (_libomp_prefix AND EXISTS ${_libomp_prefix})
        list(APPEND CMAKE_PREFIX_PATH ${_libomp_prefix})
    endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

find_package(pybind11 REQUIRED)
find_package(cereal REQUIRED)
find_package(PCGRandom REQUIRED)

# Now we get to adding our components. Let's do some global setup such as
# setting the CXX standard and the shared library details.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (LINUX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
elseif (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif ()

add_subdirectory(external/libalgebra_lite)


add_subdirectory(core)
add_subdirectory(platform)
add_subdirectory(scalars)
add_subdirectory(intervals)
add_subdirectory(algebra)
add_subdirectory(streams)
add_subdirectory(roughpy)

if (ROUGHPY_GENERATE_DEVICE_CODE)
    add_subdirectory(device)
endif ()

if (ROUGHPY_BUILD_LA_CONTEXTS)
    set(LIBALGEBRA_NO_SERIALIZATION ON CACHE INTERNAL "")
    add_subdirectory(external/libalgebra)
    add_subdirectory(la_context)
endif ()


# TODO: Maybe we should replace this with a custom install target rather than
# messing with the install directories.
#set(CMAKE_INSTALL_LIBDIR "roughpy" CACHE STRING "install library dir")
#set(CMAKE_INSTALL_BINDIR "roughpy" CACHE STRING "install binary dir")

install(TARGETS
        RoughPy_PyModule
        EXPORT RoughPy_EXPORTS
        RUNTIME DESTINATION roughpy
        LIBRARY
        DESTINATION roughpy
        NAMELINK_SKIP
        ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
        COMPONENT Development
        EXCLUDE_FROM_ALL
        INCLUDES DESTINATION ${SKBUILD_NULL_DIR}
        COMPONENT Development
        EXCLUDE_FROM_ALL
        )

install(FILES roughpy/py.typed DESTINATION roughpy)
install(DIRECTORY roughpy
        DESTINATION .
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.pyi"
        PATTERN "src/*" EXCLUDE)

set(_runtime_deps "")
foreach (_rpy_lib IN LISTS ROUGHPY_LIBS)
    get_target_property(_lib_deps ${_rpy_lib} RUNTIME_DEPENDENCIES)
    if (_lib_deps)
        list(APPEND _runtime_deps "${_lib_deps}")
        #        install(FILES ${_lib_deps} DESTINATION roughpy)
    endif ()
endforeach ()
install(FILES ${_runtime_deps} DESTINATION roughpy)
