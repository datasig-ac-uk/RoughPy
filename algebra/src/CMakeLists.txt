

add_subdirectory(libalgebra_lite_internal)



if (ROUGHPY_BUILD_TESTS)

    add_executable(test_algebra
            ContextFixture.cpp
            ContextFixture.h
            test_lie.cpp
    )

    target_link_libraries(test_algebra PRIVATE RoughPy::Algebra GTest::gtest)

    setup_roughpy_cpp_tests(test_algebra)

endif()

