
if (GMP_FOUND)
    return()
endif()

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_package(PkgConfig QUIET)

if (PkgConfig_FOUND)
    pkg_check_modules(PC_GMP QUIET gmp IMPORTED_TARGET)
endif ()

if (PC_GMP_FOUND)
    set(GMP_INCLUDE_DIR "${PC_GMP_INCLUDE_DIRS}")
    set(GMP_LIBRARY_DIRS "${PC_GMP_LIBRARY_DIRS}")
    set(GMP_LIBRARIES "${PC_GMP_LIBRARIES}")
    set(GMP_COMPILE_OPTIONS ${PC_GMP_CFLAGS} ${PC_GMP_CFLAGS_OTHER})
    set(GMP_LINK_OPTIONS ${PC_GMP_LDFLAGS} ${PC_GMP_LDFLAGS_OTHER})

    add_library(GMP::GMP ALIAS PkgConfig::PC_GMP)
else()

    set(_gmp_lib_names gmp libgmp gmp-10 libgmp-10 mpir libmpir)

    find_library(GMP_LIBRARY_RELEASE
            NAMES ${_gmp_lib_names}
            HINTS "${VCPKG_INSTALLED_DIR}"
    )

    if (WIN32)
        # VCPKG installs debug libraries in ${VCPKG_INSTALLED_DIR}/debug on Windows
        find_library(GMP_LIBRARY_DEBUG
                NAMES ${_gmp_lib_names}
                HINTS "${VCPKG_INSTALLED_DIR}/debug"
        )
    else()
        foreach(_name IN ITEMS ${_gmp_lib_names})
            list(APPEND _gmp_lib_names_debug "${_name}d")
        endforeach()

        # Almost surely, sensible platforms will have pkgconfig find the libraries
        find_library(GMP_LIBRARY_DEBUG
                NAMES ${_gmp_lib_names_debug}
                HINTS "${VCPKG_INSTALLED_DIR}"
        )
    endif()

    set(GMP_COMPILE_OPTIONS "")
    set(GMP_LINK_OPTIONS "")

    find_path(GMP_INCLUDE_DIR NAMES gmp.h HINTS "${VCPKG_INSTALLED_DIR}")


    select_library_configurations(GMP)
    find_package_handle_standard_args(GMP
            FOUND_VAR GMP_FOUND
            REQUIRED_VARS
            GMP_LIBRARY GMP_INCLUDE_DIR)

    mark_as_advanced(
            GMP_INCLUDE_DIR
            GMP_LIBRARIES
            GMP_LIBRARY_DIRS
            GMP_COMPILE_OPTIONS
            GMP_LINK_OPTIONS
    )

    if (GMP_FOUND AND NOT TARGET GMP::GMP)
        add_library(GMP::GMP UNKNOWN IMPORTED GLOBAL)
        set_target_properties(GMP::GMP PROPERTIES
                IMPORTED_LOCATION "${GMP_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
                IMPORTED_IMPLIB "${GMP_LIBRARY}"
        )
        #    if (GMP_COMPILE_OPTIONS)
        #        set_target_properties(GMP::GMP PROPERTIES
        #                INTERFACE_COMPILE_OPTIONS "${GMP_COMPILE_OPTIONS}"
        #        )
        #    endif()
        #    if(GMP_LINK_OPTIONS)
        #        set_target_properties(GMP::GMP PROPERTIES
        #                INTERFACE_LINK_OPTIONS "${GMP_LINK_OPTIONS}"
        #        )
        #    endif()
        #
        #    if (GMP_LIBRARY_RELEASE)
        #        set_property(TARGET GMP::GMP APPEND PROPERTY
        #            IMPORTED_CONFIGURATIONS RELEASE
        #        )
        #        set_target_properties(GMP::GMP PROPERTIES
        #                IMPORTED_LOCATION_RELEASE "${GMP_LIBRARY_RELEASE}"
        #        )
        #    endif()
        #    if (GMP_LIBRARY_DEBUG)
        #        set_property(TARGET GMP::GMP APPEND PROPERTY
        #                IMPORTED_CONFIGURATIONS DEBUG
        #        )
        #        set_target_properties(GMP::GMP PROPERTIES
        #                IMPORTED_LOCATION_DEBUG "${GMP_LIBRARY_DEBUG}"
        #        )
        #    endif()


    endif()
endif ()

