
include(FindPackageHandleStandardArgs)

find_path(PCGRandom_INCLUDE_DIRS NAMES pcg_random.hpp)

find_package_handle_standard_args(PCGRandom
        FOUND_VAR PCGRandom_FOUND
        REQUIRED_VARS PCGRandom_INCLUDE_DIRS
        )


if (PCGRandom_FOUND AND NOT TARGET PCGRandom::pcg_random)

    add_library(PCGRandom::pcg_random IMPORTED INTERFACE GLOBAL)
    set_target_properties(PCGRandom::pcg_random PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PCGRandom_INCLUDE_DIRS}"
            )

endif ()
