
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
if (NOT PROJECT_IS_TOP_LEVEL OR NOT ROUGHPY_BUILD_TESTS)
    return()
endif ()

IF (CMAKE_GENERATOR MATCHES "Makefiles|Ninja")
    option(ROUGHPY_ENABLE_IWYU "Enable include-what-you-use" OFF)

    if (ROUGHPY_ENABLE_IWYU)

        find_program(IWYU_EXECUTABLE include-what-you-use REQUIRED)
        set(CMAKE_C_INCLUDE_WHAT_YOU_USE
                ${IWYU_EXECUTABLE}
                -Xiwyu --mapping_file=${CMAKE_SOURCE_DIR}/tools/IWYU/roughpy.imp
                #                        -Xiwyu --error
                CACHE STRING "Include what you use command"
        )
        set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE
                ${IWYU_EXECUTABLE}
                        -Xiwyu --mapping_file=${CMAKE_SOURCE_DIR}/tools/IWYU/roughpy.imp
#                        -Xiwyu --error
                CACHE STRING "Include what you use command"
        )

    endif ()
endif ()


cmake_dependent_option(ROUGHPY_PROFILE_BUILD
        "Build with -ftime-trace when using Clang"
        OFF
        "CMAKE_CXX_COMPILER_ID MATCHES \"Clang\";NOT APPLE"
        OFF)
if (ROUGHPY_PROFILE_BUILD)
    add_compile_options("-ftime-trace")
endif()



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
    add_compile_definitions($<$<BOOL:${ROUGHPY_ENABLE_UBSAN}>:RPY_USING_UBSAN=1>)
endif ()


# Professional CMake 19th ed. pp. 465-469
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 --coverage")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        string(APPEND CMAKE_CXX_FLAGS_COVERAGE " -fprofile-abs-path")
    endif ()

    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage")
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "--coverage")
    set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE "--coverage")


    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        execute_process(
                COMMAND xcrun --find gcov
                OUTPUT_VARIABLE GCOV_EXECUTABLE
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_program(LLVM_COV_EXECUTABLE llvm-cov REQUIRED)
        file(CREATE_LINK ${LLVM_COV_EXECUTABLE} ${CMAKE_BINARY_DIR}/gcov SYMBOLIC)
        set(GCOV_EXECUTABLE "${LLVM_COV_EXECUTABLE} gcov")
    else ()
        find_program(GCOV_EXECUTABLE gcov REQUIRED)
    endif ()

    configure_file(${CMAKE_CURRENT_LIST_DIR}/gcovr.cfg.in ${CMAKE_BINARY_DIR}/gcovr.cfg @ONLY)

    add_custom_target(process_coverage
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running gcovr to process coverage results"
            COMMAND ${GCOVR_EXECUTABLE} --config gcovr.cfg .
    )

endif ()

