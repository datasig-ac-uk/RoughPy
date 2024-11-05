cmake_minimum_required(VERSION 3.28)

include(CMakeDependentOption)
include(CheckIncludeFileCXX)


option(ROUGHPY_NO_VCPKG "Do not force install VCPKG if is not already set" OFF)
option(ROUGHPY_BUILD_TESTS "Build C++ tests for RoughPy" OFF)
option(ROUGHPY_BUILD_DOCS "Build the documentation" OFF)

option(ROUGHPY_USE_CCACHE "Use compiler cache to speed up repeated builds" ON)

option(ROUGHPY_BUILD_PYLIB "Build the Python library for RoughPy" ON)
option(ROUGHPY_LINK_NUMPY "Link with Numpy library for array handling" ${ROUGHPY_BUILD_PYLIB})
option(ROUGHPY_BUILD_PYMODULE_INPLACE "Build the pymodule in the project roughpy directory" OFF)

option(ROUGHPY_GENERATE_DEVICE_CODE "Generate code for objects on devices" OFF)
option(ROUGHPY_DISABLE_BLAS "Disable linking to blas/lapack" ON)
cmake_dependent_option(ROUGHPY_PREFER_ACCELERATE
        "Prefer Accelerate framework on MacOS always" OFF APPLE OFF)



if (NOT CMAKE_TOOLCHAIN_FILE AND NOT ROUGHPY_NO_VCPKG)
    cmake_path(APPEND CMAKE_CURRENT_LIST_DIR "tools" "vcpkg"
            OUTPUT_VARIABLE _vcpkg_dir)

    if (NOT EXISTS _vcpkg_dir)
        find_package(Git REQUIRED)
        execute_process(COMMAND
                "${GIT_EXECUTABLE}" "clone" "https://github.com/Microsoft/vcpkg.git" "${_vcpkg_dir}"
        )
    endif()

    cmake_path(APPEND _vcpkg_dir "scripts" "buildsystems" "vcpkg.cmake" OUTPUT_VARIABLE _vcpkg_file)
    message(STATUS "installed VCPKG in ${_vcpkg_dir}")
    set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_file}" CACHE INTERNAL "")
endif()
message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")



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



include(GenerateExportHeader)

# TODO: Move this down once the helpers is cleaned out
if (ROUGHPY_BUILD_TESTS)
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
    if (DEFINED ROUGHPY_PYTHON_VENV_DIR)
        list(PREPEND CMAKE_PREFIX_PATH "${ROUGHPY_PYTHON_VENV_DIR}")
    endif()
endif ()


# At minimum we need Interpreter and Development.Module in order to build a
# Python extension module.
set(PYTHON_COMPONENTS_NEEDED Interpreter Development.Module)
#if (ROUGHPY_LINK_NUMPY)
#    list(APPEND PYTHON_COMPONENTS_NEEDED NumPy)
#endif ()

find_package(Python 3.8 REQUIRED COMPONENTS ${PYTHON_COMPONENTS_NEEDED})

if (ROUGHPY_LINK_NUMPY)
    find_package(Python 3.8 QUIET COMPONENTS NumPy)

    if (NOT TARGET Python::NumPY)
        message(STATUS "Installing NumPy")
        execute_process(COMMAND "${Python_EXECUTABLE} -m pip install numpy" COMMAND_ECHO STDOUT)
        find_package(Python REQUIRED COMPONENTS NumPy)
    endif()

endif()


if(NOT DEFINED pybind11_ROOT)
    execute_process(COMMAND "${Python_EXECUTABLE}" "-m" "pip" "install" "pybind11"
                    COMMAND_ECHO STDOUT
                    RESULT_VARIABLE _pybind11_install_result
                    OUTPUT_VARIABLE _pybind11_install_message
    )

    if (NOT _pybind11_install_result AND NOT _pybind11_install_message MATCHES "Requirement already satisfied: pybind11")
        message(FATAL_ERROR "could not install pybind11")
    endif()

    unset(_pybind11_install_result)
    execute_process(COMMAND
            "${Python_EXECUTABLE}" "-m" "pybind11" "--cmakedir"
            COMMAND_ECHO STDOUT
            RESULT_VARIABLE _python_pybind11_dir_found
            OUTPUT_VARIABLE _python_pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (_python_pybind11_dir_found EQUAL 0)
        message(STATUS "Adding Pybind11 cmake dir ${_python_pybind11_dir}")
        set(pybind11_ROOT "${_python_pybind11_dir}" CACHE INTERNAL "")
        list(APPEND CMAKE_PREFIX_PATH "${_python_pybind11_dir}")
    else()
        message(FATAL_ERROR "Could not add pybind11 dir: ${_python_pybind11_dir_found} && ${_python_pybind11_dir}")
    endif ()
endif ()

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
    smart_ptr
    type_traits
    url
    uuid
)


find_package(Eigen3 CONFIG REQUIRED)
find_package(SndFile CONFIG CONFIG REQUIRED)
find_package(OpenCL CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(cereal CONFIG REQUIRED)
find_package(PCGRandom REQUIRED)
find_package(GMP REQUIRED)

message(STATUS "Target architecture ${RPY_ARCH}")







if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path" CACHE INTERNAL "")
elseif (NOT WIN32)
    set(CMAKE_INSTALL_RPATH "$ORIGIN" CACHE INTERNAL "")
endif ()



set(LIBALGEBRA_LITE_BUILD_TESTS OFF CACHE INTERNAL "")
set(LIBALGEBRA_LITE_RATIONAL_COEFFS OFF CACHE INTERNAL "")
add_subdirectory(external/libalgebra_lite)


add_subdirectory(core)
add_subdirectory(platform)
add_subdirectory(scalars)
add_subdirectory(intervals)
add_subdirectory(algebra)
add_subdirectory(streams)

if (ROUGHPY_BUILD_PYLIB)
    add_subdirectory(roughpy)
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
        FRAMEWORK DESTINATION roughpy
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
#install(FILES ${_runtime_deps} DESTINATION roughpy)
