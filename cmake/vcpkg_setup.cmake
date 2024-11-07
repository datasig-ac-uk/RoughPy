
include_guard()

if (ROUGHPY_NO_VCPKG OR "${CMAKE_TOOLCHAIN_FILE}" MATCHES "vcpkg.cmake")
    # Do not use vcpkg, or vcpkg toolchain already set up
    message(DEBUG "Not using vcpkg or toolchain already set")
    return()
endif()


function(_setup_vcpkg _path)
    find_package(GIT REQUIRED)
    message(STATUS "Cloning vcpkg into \"${_real_path}\"")
    execute_process(COMMAND
            ${GIT_EXECUTABLE} clone
            "https://github.com/Microsoft/vcpkg.git"
            "${_real_path}"
            RESULT_VARIABLE _success
            ERROR_VARIABLE _err_msg
            OUTPUT_QUIET
    )

    if (NOT _success)
        message(FATAL_ERROR "Failed to clone vcpkg:\n${_err_msg}")
    endif()
endfunction()

function(_find_or_install_vcpkg _toolchain_location_var)

    cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR "tools" "vcpkg" _vcpkg_in_source_root)

    # First check if VCPKG_ROOT is set as a CMake variable or in the environment
    if (DEFINED VCPKG_ROOT)
        cmake_path(ABSOLUTE_PATH VCPKG_ROOT NORMALIZE _vcpkg_root)
    elseif(DEFINED ENV{VCPKG_ROOT})
        cmake_path(ABSOLUTE_PATH ENV{VCPKG_ROOT} NORMALIZE _vcpkg_root)
    else()
        set(_vcpkg_root ${_vcpkg_in_source_root})
    endif()
    message(DEBUG "Looking for vcpkg in ${_vcpkg_root}")

    cmake_path(APPEND _vcpkg_root "scripts" "buildsystems" "vcpkg.cmake" _toolchain_file)

    if (NOT EXISTS ${_toolchain_file})
        if (ROUGHPY_NO_INSTALL_VCPKG)
            message(FATAL_ERROR "vcpkg is requested but not found, and automatic installation is disabled")
        else()
            cmake_path(NATIVE_PATH _vcpkg_in_source_root _vcpkg_install_dir)
            _setup_vcpkg(${_vcpkg_install_dir})

            # If successful, reset the toolchain file to point to the correct file
            cmake_path(APPEND _vcpkg_in_source_root "scripts" "buildsystems" "vcpkg.cmake" _toolchain_file)
        endif()
    endif()

    message(DEBUG "Using vcpkg toolchain file ${_toolchain_file}")
    set(${_toolchain_location_var} ${_toolchain_file} PARENT_SCOPE)
endfunction()


if (DEFINED CMAKE_TOOLCHAIN_FILE)
    # An already defined toolchain defined (that is not vcpkg) needs
    # to be chain-loaded
    message(STATUS "Chain loading ${CMAKE_TOOLCHAIN_FILE}")
    set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE}
            CACHE FILEPATH "Chain loaded toolchain file")
endif()

_find_or_install_vcpkg(_vcpkg_toolchain)
set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_toolchain}"
        CACHE FILEPATH "vcpkg toolchain file location" FORCE)


message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
