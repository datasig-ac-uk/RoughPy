cmake_minimum_required(VERSION 3.21)

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

set(CMAKE_INSTALL_LIBDIR "roughpy" CACHE STRING "install library dir")
set(CMAKE_INSTALL_BINDIR "roughpy" CACHE STRING "install binary dir")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (LINUX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
elseif (APPLE)
    execute_process(COMMAND "brew" "--prefix"
            RESULT_VARIABLE _brew_prefix_found
            OUTPUT_VARIABLE _brew_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    message(STATUS "Adding brew prefix: ${_brew_prefix}")
    if (_brew_prefix_found)
        list(APPEND CMAKE_PREFIX_PATH "${_brew_prefix}")
    endif ()


    #    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_MACOSX_RPATH ON)
    #    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
    #    set(CMAKE_INSTALL_RPATH @loader_path)
    #    set(CMAKE_INSTALL_NAME_DIR @rpath)
endif ()

option(ROUGHPY_BUILD_LA_CONTEXTS "Build the collection of libalgebra contexts" OFF)
option(ROUGHPY_BUILD_TESTS "Build C++ tests for RoughPy" ON)
option(ROUGHPY_BUILD_PYMODULE_INPLACE "Buildg the pymodule in the project roughpy directory" OFF)

if (MSVC)
    #    add_compile_options(/permissive-)
endif ()

find_package(GTest CONFIG QUIET)
if (ROUGHPY_BUILD_TESTS AND GTest_FOUND)
    find_package(GTest CONFIG REQUIRED)

    if (NOT TARGET GTest::gtest)
        message(FATAL_ERROR "GTest::gtest target not defined")
    endif ()

    enable_testing()
else ()
    set(ROUGHPY_BUILD_TESTS OFF CACHE INTERNAL "")
endif ()

include(cmake/roughpy_helpers.cmake)


option(ROUGHPY_LINK_NUMPY "Link with Numpy library for array handling" ON)
option(ROUGHPY_GENERATE_DEVICE_CODE "Generate code for objects on devices" OFF)


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
    set(Python_FIND_VIRTUALENV FIRST)
endif ()

if (ROUGHPY_LINK_NUMPY)
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
else ()
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
endif ()

if (DEFINED ENV{VIRTUAL_ENV})
    # Put venv/lib on the prefix path so we can find
    # a pip installed MKL
    message(STATUS "Adding python virtual environment to path")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}/lib")
endif ()

# Before we get too far, let's make use of Python's import
# system to make sure MKL can be found if it was pip installed
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
if (RPY_ARCH MATCHES "[xX](86(_64)?|64)|[aA][mM][dD]64")
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

        #        if (DEFINED MKL_OMP_LIB AND MKL_OMP_LIB)
        #            foreach (LANG IN ITEMS C CXX)
        #                set(OpenMP_${LANG}_LIB_NAMES ${MKL_OMP_LIB} CACHE STRING "Intel OpenMP runtime library")
        #            endforeach ()
        #        endif ()
        if (DEFINED MKL_OMP_LIB AND MKL_OMP_LIB)
            foreach (LANG IN ITEMS C CXX)
                if (APPLE)
                    set(OpenMP_${LANG}_FLAGS "-XPreprocessor -fopenmp=${MKL_OMP_LIB}")
                    set(OpenMP_${LANG}_LIB_NAMES "${MKL_OMP_LIB}" CACHE STRING "libomp location for OpenMP")
                endif ()
            endforeach ()
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

find_package(OpenMP REQUIRED COMPONENTS CXX)

find_package(pybind11 REQUIRED)
find_package(cereal REQUIRED)
find_package(PCGRandom REQUIRED)


add_subdirectory(external/libalgebra_lite)
#set_target_properties(Libalgebra_lite PROPERTIES NO_SONAME ON)

#add_subdirectory(external/pybind11)


# Make an imported target for PCG-CPP because it is a
# makefile based project
#if(NOT TARGET PCGRandom::pcg_random)
#    add_library(PCGRandom::pcg_random IMPORTED INTERFACE)
#    target_include_directories(PCGRandom::pcg_random INTERFACE
#            external/pcg-cpp/include)
#endif()


#add_subdirectory(external/recombine)

set(LIBALGEBRA_NO_SERIALIZATION ON CACHE INTERNAL "")
add_subdirectory(external/libalgebra)

#add_subdirectory(external/csv-parser)
# The csv-parser CMakeLists.txt leaves much to be desired
# Let's fix some of the problems now
#if(NOT TARGET csv::csv)
#    add_library(csv::csv ALIAS csv)
#    target_include_directories(csv INTERFACE
#            external/csv-parser/include)
#    set_target_properties(csv PROPERTIES POSITION_INDEPENDENT_CODE ON)
#endif()

#set(BUILD_DOC OFF CACHE INTERNAL "disable cereal docs")
#set(BUILD_SANDBOX OFF CACHE INTERNAL "disable cereal sandbox examples")
#set(SKIP_PERFORMANCE_COMPARISON ON CACHE INTERNAL "disable building cereal performance tests")
#add_subdirectory(external/cereal)

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
    add_subdirectory(la_context)
endif ()


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

#
#install(EXPORT RoughPy_EXPORTS
#        DESTINATION roughpy
#        NAMESPACE RoughPy
#        FILE RoughPy.cmake
#            COMPONENT Development
#            EXCLUDE_FROM_ALL
#        )
#
#
#if (TARGET MKL::MKL AND DEFINED MKL_THREAD_LIB)
#    install(FILES ${MKL_THREAD_LIB} DESTINATION roughpy)
#endif ()
