cmake_minimum_required(VERSION 3.28)

include(CMakeDependentOption)


option(ROUGHPY_NO_VCPKG "Do not use VCPKG if is not already set" OFF)
cmake_dependent_option(
        ROUGHPY_NO_INSTALL_VCPKG
        "Do not install VCPKG if it is not found"
        OFF
        "NOT ROUGHPY_NO_VCPKG"
        OFF
)

# Control the parts of RoughPy that are built
option(ROUGHPY_BUILD_TESTS "Build C++ tests for RoughPy" OFF)
option(ROUGHPY_BUILD_DOCS "Build the documentation" OFF)

option(ROUGHPY_EXPERIMENTAL "Allow building experimental features" OFF)

# Modify the build process
option(ROUGHPY_USE_CCACHE "Use compiler cache to speed up repeated builds" ON)

# Control the building and properties of the Python module
option(ROUGHPY_BUILD_PYLIB "Build the Python library for RoughPy" ON)
option(ROUGHPY_LINK_NUMPY "Link with Numpy library for array handling" ${ROUGHPY_BUILD_PYLIB})

# For developer options, see cmake/developer_options_setup.cmake

# Currently unused
option(ROUGHPY_GENERATE_DEVICE_CODE "Generate code for objects on devices" OFF)
option(ROUGHPY_DISABLE_BLAS "Disable linking to blas/lapack" ON)


# RoughPy usually requires vcpkg to ensure all the dependencies are installed
# and available for the build system. This file sets everything up including,
# if allowed, downloading vcpkg into tools/vcpkg
include(cmake/vcpkg_setup.cmake)


# This sets the the manifest features to include GoogleTest if we're building tests
# This means we don't have to interact with vcpkg at all even when we need to build
# the test suite.
if (ROUGHPY_BUILD_TESTS)
    set(VCPKG_MANIFEST_FEATURES "tests" CACHE INTERNAL "")
endif ()



# Set the version for Roughpy.
# TODO: This needs a dramatic overhaul.
if (EXISTS "VERSION.txt")
    file(READ "VERSION.txt" _rpy_version)
    message(STATUS "Repository version ${_rpy_version}")
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _rpy_version "${_rpy_version}")
else ()
    set(_rpy_version 0.0.1)
endif ()


###############################################################################
# Everything above this line has to happen before the call to project, because
# project sets up the toolchain and vcpkg. Only things that must come before
# the call to project should be above.
###############################################################################
project(RoughPy VERSION ${_rpy_version})




# We have some custom find modules to
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)



# Generate a compilation database, which helps with language servers in some IDEs
# and text editors.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Now we get to adding our components. Let's do some global setup such as
# setting the CXX standard and the shared library details.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# I don't think we have any pure C anywhere, but we set the standard anyway
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# TODO: Handle these more delicately.
if (MSVC)
    ## Disable non-compliant behaviour on windows... Comon MS>
    add_compile_options("/permissive-")
endif()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version OUTPUT_VARIABLE stdout ERROR_QUIET)
    if ("${stdout}" MATCHES "GNU gold")
        message(STATUS "using gold linker")
        add_link_options(-fuse-ld=gold)
    endif ()
endif ()


# Standard CMake provided modules
include(CheckIncludeFileCXX)
include(GenerateExportHeader)
include(FetchContent)

# TODO: Move this down once the helpers is cleaned out
if (ROUGHPY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(testing)
endif()

# If ROUGHPY_USE_CCACHE is set, setup the project to use ccache to speed up builds
include(cmake/setup_ccache.cmake)

# Scikit-Build-Core requires that the built artifacts are installed into specific
# directories provided as cache variables during the configuration step. This file
# contains a little logic that makes sure the install locations are set accordingly
# and otherwise sets using the default GNUInstallDirs locations.
include(cmake/setup_install_dirs.cmake)

# Load the helper functions
include(cmake/roughpy_helpers.cmake)

# RoughPy components might have sub-components, defined in CMakeLists in directories
# below the main component. To help these set their include directories and link
# with other parts of the library, we set a number of variables that point to the
# component level source tree. See the documentation of the function for a list
# of the variables set and their purpose.
include(cmake/component_setup.cmake)

# When developing RoughPy, it's a good idea to build with as much diagnostic information
# as possible. To maintain platform independence, RoughPy has two options for including
# the necessary flags for compiler warnings. These options are turned into compiler
# options in this file.
include(cmake/developer_options_setup.cmake)




set(RPY_ARCH ${CMAKE_SYSTEM_PROCESSOR})
# TODO: handle this better
if (DEFINED CMAKE_CXX_COMPILER_TARGET)
    set(RPY_ARCH ${CMAKE_CXX_COMPILER_TARGET})
endif ()


# We use C++17 standard library headers. If these aren't available for some
# reason, we can fall back to Boost versions but this is obviously not desirable
check_include_file_cxx(filesystem RPY_HAS_STD_FILESYSTEM)
check_include_file_cxx(optional RPY_HAS_STD_OPTIONAL)

# find_package(Boost) is deprecated and finding the CONFIG package might pick
# up a system library and not the ones installed by vcpkg. For this reason,
# we provide a function that finds boost_{component} for each component.
# This sets the Boost::{component} targets to be used in linking as usual.
find_boost(VERSION 1.83 COMPONENTS
    headers
    align
    container
    core
    endian
    interprocess
    multiprecision
    pool
    smart_ptr
    type_traits
    url
    uuid
)


find_package(Eigen3 CONFIG REQUIRED)
find_package(SndFile CONFIG CONFIG REQUIRED)
find_package(OpenCL CONFIG REQUIRED)
find_package(cereal CONFIG REQUIRED)
find_package(PCGRandom REQUIRED)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(ctre CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

message(STATUS "Target architecture ${RPY_ARCH}")







if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path" CACHE INTERNAL "")
elseif (NOT WIN32)
    set(CMAKE_INSTALL_RPATH "$ORIGIN" CACHE INTERNAL "")
endif ()



add_subdirectory(vendored/libalgebra_lite)


add_subdirectory(core)
add_subdirectory(platform)
add_subdirectory(scalars)
add_subdirectory(intervals)
add_subdirectory(algebra)
add_subdirectory(streams)

if (ROUGHPY_BUILD_PYLIB)
    add_subdirectory(roughpy)


    install(TARGETS RoughPy_PyModule
            RUNTIME DESTINATION roughpy
            LIBRARY
            DESTINATION roughpy
            NAMELINK_SKIP
            ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
            COMPONENT Development
            EXCLUDE_FROM_ALL
            INCLUDES DESTINATION ${SKBUILD_NULL_DIR}
            COMPONENT Development
            FRAMEWORK DESTINATION roughpy
            EXCLUDE_FROM_ALL
    )

    install(DIRECTORY roughpy
            DESTINATION .
            FILES_MATCHING
            PATTERN "py.typed"
            PATTERN "*.py"
            PATTERN "*.pyi"
            PATTERN "src/*" EXCLUDE)

endif ()


if (ROUGHPY_BUILD_DOCS)
    # The docs subdirectory has it's own find
    add_subdirectory(docs)
endif()

install(TARGETS
        RoughPy_Platform
        RoughPy_Intervals
        RoughPy_Scalars
        RoughPy_Algebra
        RoughPy_Streams
        Libalgebra_lite
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
        FRAMEWORK DESTINATION roughpy
    EXCLUDE_FROM_ALL
)


