
include_guard()


if (MSVC)
    if (ROUGHPY_ENABLE_EXTRA_WARNINGS)
        add_compile_options("/W4")
    elseif(ROUGHPY_ENABLE_ALL_WARNINGS)
        add_compile_options("/W3")
    endif()
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU|CLANG")
    if (ROUGHPY_ENABLE_ALL_WARNINGS OR ROUGHPY_ENABLE_EXTRA_WARNINGS)
        add_compile_options("-Wall")
    endif()
    if (ROUGHPY_ENABLE_EXTRA_WARNINGS)
        add_compile_options("-Wextra")
    endif()
endif()