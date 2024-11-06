
enable_testing()

# find_package imports things locally to the current CMakeLists.txt, so we need the
# GLOBAL flag to make it available to the rest of the project.
find_package(GTest CONFIG REQUIRED GLOBAL)

if (NOT TARGET GTest::gtest)
    message(FATAL_ERROR "GTest::gtest target not defined")
endif ()

include(GoogleTest)





# TODO: Add a test runner similar to gtest_main that we can link into
#       each of our test executables.










function(setup_roughpy_cpp_tests _test_name)


    target_link_libraries(${_test_name} PRIVATE GTest::gtest_main)

    if (WIN32)


    endif()


    gtest_discover_tests(${_test_name})
endfunction()