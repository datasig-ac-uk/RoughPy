
include_guard()


option(ROUGHPY_ENABLE_ALL_WARNINGS "Enable -Wall or equivalent" OFF)
option(ROUGHPY_ENABLE_EXTRA_WARNINGS "Enable -Wextra or equivalent" OFF)

if (MSVC)
    if (ROUGHPY_ENABLE_EXTRA_WARNINGS)
        add_compile_options("/W4")
    elseif (ROUGHPY_ENABLE_ALL_WARNINGS)
        add_compile_options("/W3")
    endif ()
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU|Clang")
    if (ROUGHPY_ENABLE_ALL_WARNINGS OR ROUGHPY_ENABLE_EXTRA_WARNINGS)
        add_compile_options("-Wall")
    endif ()
    if (ROUGHPY_ENABLE_EXTRA_WARNINGS)
        add_compile_options("-Wextra")
    endif ()
endif ()

# Sanitizers are only supported if the RoughPy is the top-level build
if (NOT PROJECT_IS_TOP_LEVEL)
    return()
endif ()

# Essentially from Professional CMake 19th Edition pp 713-715
option(ROUGHPY_ROUGHPY_ENABLE_ASAN "Enable Address sanitizer" OFF)


if (MSVC)
    if (ROUGHPY_ROUGHPY_ENABLE_ASAN)
        string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
        string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        add_compile_options(/fsanitize=address /fsantize-address-use-after-return)
    endif ()
elseif (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    option(ROUGHPY_ENABLE_LSAN "Enable LeakSanitizer" OFF)
    option(ROUGHPY_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
    option(ROUGHPY_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
    if (NOT APPLE)
        option(ROUGHPY_ENABLE_MSAN "Enable MemorySanitizer" OFF)
    endif ()

    if ((ROUGHPY_ENABLE_ASAN AND (ROUGHPY_ENABLE_TSAN OR ROUGHPY_ENABLE_MSAN)) OR
    (ROUGHPY_ENABLE_LSAN AND (ROUGHPY_ENABLE_TSAN OR ROUGHPY_ENABLE_MSAN)) OR
    (ROUGHPY_ENABLE_TSAN AND ROUGHPY_ENABLE_MSAN))
        message(FATAL_ERROR
                "Invalid sanitizer combination:\n"
                " ROUGHPY_ENABLE_ASAN: ${ROUGHPY_ENABLE_ASAN}\n"
                " ROUGHPY_ENABLE_LSAN: ${ROUGHPY_ENABLE_LSAN}\n"
                " ROUGHPY_ENABLE_TSAN: ${ROUGHPY_ENABLE_TSAN}\n"
                " ROUGHPY_ENABLE_MSAN: ${ROUGHPY_ENABLE_MSAN}"
        )
    endif ()

    add_compile_options(
            -fno-omit-frame-pointer
            $<$<BOOL:${ROUGHPY_ENABLE_ASAN}>:-fsanitize=address>
            $<$<BOOL:${ROUGHPY_ENABLE_LSAN}>:-fsanitize=leak>
            $<$<BOOL:${ROUGHPY_ENABLE_MSAN}>:-fsanitize=memory>
            $<$<BOOL:${ROUGHPY_ENABLE_TSAN}>:-fsanitize=thread>
            $<$<BOOL:${ROUGHPY_ENABLE_UBSAN}>:-fsanitize=undefined>
    )
    add_link_options(
            $<$<BOOL:${ROUGHPY_ENABLE_ASAN}>:-fsanitize=address>
            $<$<BOOL:${ROUGHPY_ENABLE_LSAN}>:-fsanitize=leak>
            $<$<BOOL:${ROUGHPY_ENABLE_MSAN}>:-fsanitize=memory>
            $<$<BOOL:${ROUGHPY_ENABLE_TSAN}>:-fsanitize=thread>
            $<$<BOOL:${ROUGHPY_ENABLE_UBSAN}>:-fsanitize=undefined>
    )
endif ()

