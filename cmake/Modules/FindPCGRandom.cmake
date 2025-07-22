
include(FindPackageHandleStandardArgs)

find_path(PCGRandom_INCLUDE_DIRS NAMES pcg_random.hpp)

find_package_handle_standard_args(PCGRandom
        FOUND_VAR PCGRandom_FOUND
        REQUIRED_VARS PCGRandom_INCLUDE_DIRS
        )


if (PCGRandom_FOUND AND NOT TARGET pcg-cpp::pcg-cpp)

    add_library(pcg-cpp::pcg-cpp IMPORTED INTERFACE GLOBAL)
    set_target_properties(pcg-cpp::pcg-cpp PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PCGRandom_INCLUDE_DIRS}"
            )

endif ()
