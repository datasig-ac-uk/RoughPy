

include(FindPackageHandleStandardArgs)
mark_as_advanced(GMP_LIBRARIES GMP_INCLUDE_DIRS)

find_package(PkgConfig QUIET)

if (PKG_CONFIG_FOUND)
    pkg_check_modules(PCGMP QUIET gmp)
endif ()

message(STATUS "Looking in ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

find_Library(GMP_LIBRARIES NAMES gmp
        PATHS
        ${PCGMP_LIBRARY_DIRS}
        ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
        PATH_SUFFIXES gmp
)


find_path(GMP_INCLUDE_DIRS NAMES gmp.h
        PATHS
        ${PCGMP_INCLUDE_DIRS}
        ${VCPKG_INSTALLED_DIR})

find_package_handle_standard_args(GMP
        FOUND_VAR GMP_FOUND
        REQUIRED_VARS GMP_LIBRARIES GMP_INCLUDE_DIRS)

if (GMP_FOUND AND NOT TARGET GMP::GMP)
    add_library(GMP::GMP IMPORTED UNKNOWN GLOBAL)

    set_target_properties(GMP::GMP PROPERTIES
            IMPORTED_LOCATION "${GMP_LIBRARIES}"
            IMPORTED_IMPLIB "${GMP_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIRS}")

endif ()
