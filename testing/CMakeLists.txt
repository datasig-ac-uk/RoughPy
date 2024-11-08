# find_package imports things locally to the current CMakeLists.txt, so we need the
# GLOBAL flag to make it available to the rest of the project.
find_package(GTest CONFIG REQUIRED GLOBAL)

if (NOT TARGET GTest::gtest)
    message(FATAL_ERROR "GTest::gtest target not defined")
endif ()

include(GoogleTest)

set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST CACHE INTERNAL "")



# TODO: Add a test runner similar to gtest_main that we can link into
#       each of our test executables.

add_library(RoughPy_test_harness STATIC src/main.cpp)


target_link_libraries(RoughPy_test_harness PUBLIC GTest::gtest)







function(setup_roughpy_cpp_tests _test_name)

    target_link_libraries(${_test_name} PRIVATE RoughPy_test_harness)

    if (WIN32)

        # Windows DLLs are dumb: they don't have a mechanism for locating their dependencies.
        # This is a major pain. To get around this, have to copy the runtime files from the build
        # tree into the directory alongside the test executable. There might be a better way of doing

        add_custom_command(TARGET ${_test_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${_test_name}> $<TARGET_RUNTIME_DLLS:${_test_name}>
                COMMAND_EXPAND_LISTS
        )

    endif()


    gtest_discover_tests(${_test_name})
endfunction()