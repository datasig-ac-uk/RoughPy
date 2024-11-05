
include_guard()


if (NOT ROUGHPY_USE_CCACHE)
    return()
endif()

find_program(CCACHE_EXECUTABLE ccache)
if (CCACHE_EXECUTABLE)
    # From p601 of "Professional CMake 15ed"
    set (ccache_ENV CACHE_SLOPPINESS=pch_defines,time_macros)

    foreach(lang IN ITEMS C CXX OBJC OBJCXX CUDA)
        set(CMAKE_${lang}_COMPILER_LAUNCHER
                ${CMAKE_COMMAND} -E env ${ccache_ENV} ${CCACHE_EXEXUTABLE}
        )
    endforeach()
endif()
