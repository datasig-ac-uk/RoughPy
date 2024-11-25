
if (MPFR_FOUND)
    return()
endif ()

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_package(PkgConfig QUIET)


if (PkgConfig_FOUND)
    pkg_check_modules(PC_MPFR QUIET mpfr IMPORTED_TARGET)
endif ()


if (PC_MPFR_FOUND)
    set(MPFR_INCLUDE_DIR "${PC_MPFR_INCLUDE_DIRS}")
    set(MPFR_LIBRARY_DIRS "${PC_MPFR_LIBRARY_DIRS}")
    set(MPFR_LIBRARIES "${PC_MPFR_LIBRARIES}")
    set(MPFR_COMPILE_OPTIONS "${PC_MPFR_CFLAGS} ${PC_MPFR_CFLAGS_OTHER}")
    set(MPFR_LINK_OPTIONS "${PC_MPFR_LDFLAGS} ${PC_MPFR_LDFLAGS_OTHER}")

    add_library(MPFR::MPFR ALIAS PkgConfig::PC_MPFR)
else ()
    set(_mpfr_lib_names mpfr libmpfr)

    find_library(MPFR_LIBRARY_RELEASE
            NAMES ${_mpfr_lib_names}
            HINTS "${VCPKG_INSTALLED_DIR}")

    set(_use_debug_dir OFF)
    if (DEFINED VCPKG_INSTALLED_DIR)
        cmake_path(IS_PREFIX VCPKG_INSTALLED_DIR CMAKE_BINARY_DIR OUTPUT_VARIABLE _use_debug_dir)
    endif ()

    if (WIN32 OR _use_debug_dir)
        find_library(MPFR_LIBRARY_DEBUG
                NAMES ${_mpfr_lib_names}
                HINTS "${VCPKG_INSTALLED_DIR}/debug"
        )
    else ()
        foreach (_name IN ITEMS ${_mpfr_lib_names})
            list(APPEND _mpfr_lib_names_debug "${_name}d")
        endforeach ()

        find_library(MPFR_LIBRARY_DEBUG
                NAMES ${_mpfr_lib_names}
                HINTS "${VCPKG_INSTALLED_DIR}")
    endif ()

    set(MPFR_COMPILE_OPTIONS "")
    set(MPFR_LINK_OPTIONS "")

    find_path(MPFR_INCLUDE_DIR NAMES mpfr.h HINTS "${VCPKG_INSTALLED_DIR}")

    select_library_configurations(MPFR)
    find_package_handle_standard_args(MPFR
            FOUND_VAR MPFR_FOUND
            REQUIRED_VARS MPFR_LIBRARY MPFR_INCLUDE_DIR
    )
    mark_as_advanced(
            MPFR_INCLUDE_DIR
            MPFR_LIBRARIES
            MPFR_LIBRARY_DIRS
            MPFR_COMPILE_OPTIONS
            MPFR_LINK_OPTIONS
    )

    if (MPFR_FOUND AND NOT MPFR::MPFR)
        add_library(MPFR::MPFR UNKNOWN IMPORTED GLOBAL)
        set_target_properties(MPFR::MPFR PROPERTIES
                IMPORTED_LOCATION "${MPFR_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${MPFR_INCLUDE_DIR}"
                IMPORTED_IMPLIB "${MPFR_LIBRARY}"
        )

    endif()


endif ()



