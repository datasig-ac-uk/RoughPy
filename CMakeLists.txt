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
option(ROUGHPY_BUILD_TEST_PYTHON_EMBED "Build embedded Python C++ tests" OFF)
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

if (DEFINED CONAN_COMMAND
        OR "${CMAKE_PROJECT_TOP_LEVEL_INCLUDES}" MATCHES "conan_provider.cmake"
        OR "${CMAKE_TOOLCHAIN_FILE}" MATCHES "conan_toolchain.cmake")
    set(RPY_USING_CONAN ON CACHE BOOL "Are we using the Conan package manager")
else()
    set(RPY_USING_CONAN CACHE BOOL "Are we using the Conan package manager")
endif()

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
set(CMAKE_C_STANDARD 17)
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
find_package(cereal CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(ctre CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

if (RPY_USING_CONAN)
    find_package(OpenCLICDLoader CONFIG REQUIRED)
    find_package(OpenCLHeaders CONFIG REQUIRED)
    find_package(pcg-cpp CONFIG REQUIRED)
    find_package(gmp CONFIG REQUIRED)
#    find_package(mpfr CONFIG REQUIRED)
else()
    find_package(OpenCL CONFIG REQUIRED)
    find_package(PCGRandom REQUIRED)
    find_package(GMP REQUIRED)
#    find_package(MPFR REQUIRED)
endif()

message(STATUS "Target architecture ${RPY_ARCH}")







if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path" CACHE INTERNAL "")
elseif (NOT WIN32)
    set(CMAKE_INSTALL_RPATH "$ORIGIN" CACHE INTERNAL "")
endif ()


###############################################################################
#                                    python                                   #
###############################################################################
# We need to provide some help to make sure we find the correct version of
# Python. Ideally, if we're using Scikit-Build-Core to build the library (via
# pip) and the Python executable is provided via the PYTHON_EXECUTABLE cache
# variable. In this case, make sure that this is the version of Python that gets
# found.
set(Python_FIND_VIRTUAL_ENVIRONMENT FIRST)
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
    if (DEFINED ROUGHPY_PYTHON_VENV_DIR)
        list(PREPEND CMAKE_PREFIX_PATH "${ROUGHPY_PYTHON_VENV_DIR}")
    endif()
endif ()


set(_python_components Interpreter Development.Module)
if (ROUGHPY_LINK_NUMPY)
    list(APPEND _python_components NumPy)
endif ()

if (ROUGHPY_BUILD_TEST_PYTHON_EMBED)
    # Required to link test_python_embed
    list(APPEND _python_components Development.Embed)
endif()


find_package(Python 3.8 REQUIRED COMPONENTS ${_python_components})
unset(_python_components)

add_subdirectory(vendored/libalgebra_lite)

add_subdirectory(core)
add_subdirectory(platform)
add_subdirectory(scalars)
add_subdirectory(intervals)
add_subdirectory(algebra)
add_subdirectory(streams)


# Currently roughpy_compute requires rational coefficients from libalgebra lite, which are disabled
# in the vendored version For this reason, the roughpy_compute CMakeLists that declares all the tests
# is not used here and instead we just declare the basic interface target here separately.
#
# In any case, this reinforces the fact that roughpy_compute is really a standalone library that might
# not always reside in roughpy proper
#add_subdirectory(roughpy_compute)

add_library(RoughPy_Compute INTERFACE
        roughpy_compute/common/architecture.hpp
        roughpy_compute/common/basis.hpp
        roughpy_compute/common/cache_array.hpp
        roughpy_compute/common/operations.hpp
        roughpy_compute/common/sparse_matrix.hpp
        roughpy_compute/dense/basic/apply_sparse_linear_map.hpp
        roughpy_compute/dense/basic/free_tensor_antipode.hpp
        roughpy_compute/dense/basic/free_tensor_fma.hpp
        roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp
        roughpy_compute/dense/basic/vector_addition.hpp
        roughpy_compute/dense/basic/vector_inplace_addition.hpp
        roughpy_compute/dense/basic/vector_inplace_scalar_multiply.hpp
        roughpy_compute/dense/basic/vector_scalar_multiply.hpp
        roughpy_compute/dense/intermediate/free_tensor_exp.hpp
        roughpy_compute/dense/intermediate/free_tensor_fmexp.hpp
        roughpy_compute/dense/intermediate/free_tensor_log.hpp
        roughpy_compute/dense/views.hpp
        roughpy_compute/common/lie_to_tensor.hpp
        roughpy_compute/common/scalars.hpp
)
target_include_directories(RoughPy_Compute INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)



if (ROUGHPY_BUILD_PYLIB)
    add_library(PythonCAPICompat INTERFACE)
    set_target_properties(PythonCAPICompat PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/vendored/pythoncapi_compat/"
    )

    add_subdirectory(pycore)
    add_subdirectory(roughpy)


    if (NOT DEFINED SKBUILD_NULL_DIR)
        set(SKBUILD_NULL_DIR roughpy/null)
    endif()

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

    install(TARGETS RoughPyComputeModule
            RUNTIME DESTINATION roughpy/compute
            LIBRARY DESTINATION roughpy/compute
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
            PATTERN "compute/*.pyi?"
            PATTERN "src" EXCLUDE
            PATTERN "compute/_src" EXCLUDE
            PATTERN "ocl_kernels" EXCLUDE
    )


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


