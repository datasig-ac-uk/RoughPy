

include(FindPackageHandleStandardArgs)


find_package(PkgConfig QUIET)

if (PKG_CONFIG_FOUND)

    pkg_check_modules(GMP IMPORTED_TARGET)

    if (TARGET PkgConfig::GMP)
        add_library(GMP::GMP ALIAS PkgConfig::GMP GLOBAL)
        return()
    endif ()
endif ()


find_Library(GMP_LIBRARIES NAMES gmp HINTS ${VCPKG_INSTALLED_DIR})
find_file(GMP_INCLUDE_DIRS NAMES gmp.h HINTS ${VCPKG_INSTALLED_DIR})


find_package_handle_standard_args(GMP FOUND_VAR GMP_FOUND REQUIRED_VARS GMP_LIBRARIES GMP_INCLUDE_DIRS)

if (GMP_FOUND AND NOT TARGET GMP::GMP)
    add_library(GMP::GMP IMPORTED UNKNOWN GLOBAL)

    set_target_properties(GMP::GMP PROPERTIES
            IMPORTED_LOCATION "${GMP_LIBRARIES}"
            IMPORTED_IMPLIB "${GMP_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIRS}")

endif ()
