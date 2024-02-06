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


include(CMakeDependentOption)

option(ROUGHPY_BUILD_COMPONENT_LIBS "Build the component libraries for RoughPy" ON)
option(ROUGHPY_ENABLE_DBG_ASSERT "enable debug assertions at runtime" OFF)
option(ROUGHPY_BUILD_LA_CONTEXTS "Build the collection of libalgebra contexts" OFF)
option(ROUGHPY_BUILD_TESTS "Build C++ tests for RoughPy" ON)
option(ROUGHPY_BUILD_PYMODULE_INPLACE "Build the pymodule in the project roughpy directory" OFF)
option(ROUGHPY_LINK_NUMPY "Link with Numpy library for array handling" ON)
option(ROUGHPY_GENERATE_DEVICE_CODE "Generate code for objects on devices" OFF)
option(ROUGHPY_DISABLE_BLAS "Disable linking to blas/lapack" ON)
cmake_dependent_option(ROUGHPY_PREFER_ACCELERATE
    "Prefer Accelerate framework on MacOS always" OFF APPLE OFF)
option(ROUGHPY_BUILD_PYLIB "Build the Python library for RoughPy" ON)

if (ROUGHPY_BUILD_TESTS)
    set(VCPKG_MANIFEST_FEATURES "tests" CACHE INTERNAL "")
endif ()

project(RoughPy VERSION ${_rpy_version} LANGUAGES CXX)

# Now we get to adding our components. Let's do some global setup such as
# setting the CXX standard and the shared library details.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckIncludeFileCXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(RPY_ARCH ${CMAKE_SYSTEM_PROCESSOR})
# TODO: handle this better
if (DEFINED CMAKE_CXX_COMPILER_TARGET)
    set(RPY_ARCH ${CMAKE_CXX_COMPILER_TARGET})
endif ()


# Check of the gold linker is available on GNU. Gold is much faster than the
# standard GCC linker.
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version OUTPUT_VARIABLE stdout ERROR_QUIET)
    if ("${stdout}" MATCHES "GNU gold")
        message(STATUS "using gold linker")
        add_link_options(-fuse-ld=gold)
    endif ()
endif ()

# If testing is enabled, find GTest to make sure the tests can be
# successfully built.
if (ROUGHPY_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)

    if (NOT TARGET GTest::gtest)
        message(FATAL_ERROR "GTest::gtest target not defined")
    endif ()
    enable_testing()
endif ()

# At the moment, we only need Python to build the Python module.
# However, because the virtual environment for dev environments is typically
# at the root of the project, we need the detection to start here, otherwise
# it can't find the virtual environment without help.
if (ROUGHPY_BUILD_PYLIB)
    # We need to provide some help to make sure we find the correct version of
    # Python. Ideally, if we're using Scikit-Build-Core to build the library (via
    # pip) and the Python executable is provided via the PYTHON_EXECUTABLE cache
    # variable. In this case, make sure that this is the version of Python that gets
    # found.
    if (NOT PYTHON_FOUND AND SKBUILD AND PYTHON_EXECUTABLE)
        cmake_path(GET PYTHON_EXECUTABLE PARENT_PATH _sk_env_dir)
        message(STATUS "SKBuild environment: ${_sk_env_dir}")

        # The following variables might have been set by SKBuild, but they cause
        # problems if we don't attempt to find the library associated (which we
        # don't need).
        unset(Python_LIBRARY CACHE)
        unset(PYTHON_LIBRARY CACHE)
        unset(Python3_LIBRARY CACHE)

        unset(_sk_env_dir)
    else ()
        set(Python_FIND_VIRTUALENV FIRST)

        if (DEFINED ROUGHPY_PYTHON_VENV_DIR)
            # We might want to customise the virtual environment we use to build
            # RoughPy (e.g. for debug builds). For this reason, set the virtual
            # environment ENV variable to this directory, if it exists.
            set(ENV{VIRTUAL_ENV} "${ROUGHPY_PYTHON_VENV_DIR}")
        endif ()
    endif ()

    set(RPY_PYTHON_COMPONENTS_NEEDED Interpreter Development.Module)
    if (ROUGHPY_LINK_NUMPY)
        list(APPEND RPY_PYTHON_COMPONENTS_NEEDED NumPy)
    endif ()

    find_package(Python 3.8 REQUIRED COMPONENTS ${RPY_PYTHON_COMPONENTS_NEEDED})
endif ()


# Load the helper functions
include(cmake/roughpy_helpers.cmake)


# We use C++17 standard library headers. If these aren't available for some
# reason, we can fall back to Boost versions but this is obviously not desirable
check_include_file_cxx(filesystem RPY_HAS_STD_FILESYSTEM)
check_include_file_cxx(optional RPY_HAS_STD_OPTIONAL)


set(Boost_NO_WARN_NEW_VERSIONS ON)
set(RPY_BOOST_VERSION 1.81)
set(RPY_BOOST_COMPONENTS url system)
if (NOT RPY_HAS_STD_FILESYSTEM)
    set(RPY_BOOST_COMPONENTS ${RPY_BOOST_COMPONENTS} filesystem)
endif ()

# All the components need the Boost headers. The other components are debatable
find_package(Boost ${RPY_BOOST_VERSION} REQUIRED COMPONENTS ${RPY_BOOST_COMPONENTS})


find_package(Eigen3 CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP QUIET IMPORTED_TARGET gmp)
if (TARGET PkgConfig::GMP)
    add_library(GMP::GMP ALIAS PkgConfig::GMP)
endif ()

# This package is c++17 so cannot be used if changing to a lower standard
# find_package(tomlplusplus CONFIG REQUIRED)

# All the components need cereal, so keep it here.
find_package(cereal REQUIRED)
#find_package(Libalgebra_lite REQUIRED)



if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path" CACHE INTERNAL "")
elseif (NOT WIN32)
    set(CMAKE_INSTALL_RPATH "$ORIGIN" CACHE INTERNAL "")
endif ()


if (ROUGHPY_BUILD_COMPONENT_LIBS)
    set(LIBALGEBRA_LITE_BUILD_TESTS OFF CACHE INTERNAL "")
    set(LIBALGEBRA_LITE_RATIONAL_COEFFS OFF CACHE INTERNAL "")
    add_subdirectory(external/libalgebra_lite)
    add_subdirectory(core)
    add_subdirectory(platform)
    add_subdirectory(scalars)
    add_subdirectory(intervals)
    add_subdirectory(algebra)
    add_subdirectory(streams)
else ()
    find_package(RoughPy CONFIG REQUIRED)
endif ()

if (ROUGHPY_BUILD_PYLIB)
    add_subdirectory(roughpy)
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

if (ROUGHPY_BUILD_COMPONENT_LIBS)
    set(RPY_COMPONENT_TARGETS
        RoughPy_Core
        RoughPy_Platform
        RoughPy_Intervals
        RoughPy_Scalars
        RoughPy_Algebra
        RoughPy_Streams)

    install(TARGETS
        ${RPY_COMPONENT_TARGETS}
        EXPORT RoughPyTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )


    write_basic_package_version_file(
        RoughPyConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    configure_package_config_file(cmake/RoughPyConfig.cmake.in
            RoughPyConfig.cmake
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/roughpy/cmake
            )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/RoughPyConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/RoughPyConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RoughPy)

    install(EXPORT RoughPyTargets
        FILE RoughPyTargets.cmake
        NAMESPACE RoughPy::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RoughPy)

endif ()

#
#if (SKBUILD)
#    install(TARGETS
#        RoughPy_PyModule
#        EXPORT RoughPy_EXPORTS
#        RUNTIME DESTINATION roughpy
#        LIBRARY
#        DESTINATION roughpy
#        NAMELINK_SKIP
#        ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
#        COMPONENT Development
#        EXCLUDE_FROM_ALL
#        INCLUDES DESTINATION ${SKBUILD_NULL_DIR}
#        COMPONENT Development
#        FRAMEWORK DESTINATION roughpy
#        EXCLUDE_FROM_ALL
#    )
#
#    install(FILES roughpy/py.typed DESTINATION roughpy)
#    install(DIRECTORY roughpy
#        DESTINATION .
#        FILES_MATCHING
#        PATTERN "*.py"
#        PATTERN "*.pyi"
#        PATTERN "src/*" EXCLUDE)
#
#    set(_runtime_deps "")
#    foreach (_rpy_lib IN LISTS ROUGHPY_LIBS)
#        get_target_property(_lib_deps ${_rpy_lib} RUNTIME_DEPENDENCIES)
#        if (_lib_deps)
#            list(APPEND _runtime_deps "${_lib_deps}")
#            #        install(FILES ${_lib_deps} DESTINATION roughpy)
#        endif ()
#    endforeach ()
#    #install(FILES ${_runtime_deps} DESTINATION roughpy)
#endif ()
