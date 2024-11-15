



if(ROUGHPY_BUILD_TESTS)


    add_executable(test_core
            test_check_macros.cpp)


    target_include_directories(test_core PRIVATE ${ROUGHPY_CORE_INCLUDE})

    target_link_libraries(test_core PRIVATE RoughPy::Core GTest::gtest)

    setup_roughpy_cpp_tests(test_core)


endif()