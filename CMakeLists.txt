cmake_minimum_required(VERSION 3.21)

set(VCPKG_INSTALL_OPTIONS "--no-print-usage")
set(VCPKG_BUILD_TYPE release)
set(VCPKG_LIBRARY_LINKAGE static)

message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

project(RoughPy VERSION 0.0.1)

set(CMAKE_INSTALL_LIBDIR "roughpy" CACHE STRING "install library dir")
set(CMAKE_INSTALL_BINDIR "roughpy" CACHE STRING "install binary dir")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (LINUX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
#    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
#    set(CMAKE_MACOSX_RPATH ON)
#    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
    set(CMAKE_INSTALL_RPATH @loader_path)
    set(CMAKE_INSTALL_NAME_DIR @rpath)
endif()

option(ROUGHPY_BUILD_LA_CONTEXTS "Build the collection of libalgebra contexts" OFF)
option(ROUGHPY_BUILD_TESTS "Build C++ tests for RoughPy" ON)

find_package(GTest CONFIG QUIET)
if (ROUGHPY_BUILD_TESTS AND GTest_FOUND)
    find_package(GTest CONFIG REQUIRED)

    if(NOT TARGET GTest::gtest)
        message(FATAL_ERROR "GTest::gtest target not defined")
    endif()

    enable_testing()
else()
    set(ROUGHPY_BUILD_TESTS OFF CACHE INTERNAL "")
endif()

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
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}/lib")
endif()

# Before we get too far, let's make use of Python's import
# system to make sure MKL can be found if it was pip installed
execute_process(${Python_EXECUTABLE}
        "./tools/python-mkl-install-dir.py"
        RESULT_VARIABLE _python_mkl_dir_found
        OUTPUT_VARIABLE _python_mkl_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
if (_python_mkl_dir_found AND EXISTS "${_python_mkl_dir}")
    set(MKL_DIR "${_python_mkl_dir}")
endif()

#set_target_properties(Libalgebra_lite PROPERTIES NO_SONAME ON)

add_subdirectory(external/pybind11)
# DLPACK has an annoying feature where it tries to build a demo
# library. Disable this here
set(BUILD_MOCK OFF CACHE BOOL INTERNAL)
add_subdirectory(external/dlpack)

# Make an imported target for PCG-CPP because it is a
# makefile based project
if(NOT TARGET PCGRandom::pcg_random)
    add_library(PCGRandom::pcg_random IMPORTED INTERFACE)
    target_include_directories(PCGRandom::pcg_random INTERFACE
            external/pcg-cpp/include)
endif()


#add_subdirectory(external/recombine)

set(LIBALGEBRA_NO_SERIALIZATION ON CACHE INTERNAL "")
add_subdirectory(external/libalgebra)

add_subdirectory(external/csv-parser)
# The csv-parser CMakeLists.txt leaves much to be desired
# Let's fix some of the problems now
if(NOT TARGET csv::csv)
    add_library(csv::csv ALIAS csv)
    target_include_directories(csv INTERFACE
            external/csv-parser/include)
    set_target_properties(csv PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(BUILD_DOC OFF CACHE INTERNAL "disable cereal docs")
set(BUILD_SANDBOX OFF CACHE INTERNAL "disable cereal sandbox examples")
set(SKIP_PERFORMANCE_COMPARISON ON CACHE INTERNAL "disable building cereal performance tests")
add_subdirectory(external/cereal)

add_subdirectory(core)
add_subdirectory(platform)
add_subdirectory(scalars)
add_subdirectory(intervals)
add_subdirectory(algebra)
add_subdirectory(streams)
add_subdirectory(roughpy)

if(ROUGHPY_GENERATE_DEVICE_CODE)
    add_subdirectory(device)
endif()

if(ROUGHPY_BUILD_LA_CONTEXTS)
    add_subdirectory(la_context)
endif()





install(TARGETS
        RoughPy_Core
        RoughPy_Platform
        RoughPy_Scalars
        RoughPy_Intervals
        RoughPy_Algebra
        RoughPy_Streams
        RoughPy_PyModule
        RUNTIME DESTINATION roughpy
        LIBRARY
            DESTINATION roughpy
            NAMELINK_SKIP
        ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
        INCLUDES DESTINATION ${SKBUILD_NULL_DIR}
        )

install(FILES
        roughpy/__init__.py
        roughpy/__init__.pyi
        roughpy/py.typed
        roughpy/version.py
        DESTINATION roughpy
        )


if (TARGET MKL::MKL AND DEFINED MKL_THREAD_LIB)
    install(FILES ${MKL_THREAD_LIB} DESTINATION roughpy)
endif()
