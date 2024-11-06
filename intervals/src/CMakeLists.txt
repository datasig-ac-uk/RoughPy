
if (ROUGHPY_BUILD_TESTS)

    add_executable(test_intervals
            test_dyadic.cpp
            test_dyadic_intervals.cpp
            test_real_interval.cpp
            test_partition.cpp
    )
    target_link_libraries(test_intervals PRIVATE RoughPy::Intervals GTest::gtest)

    setup_roughpy_cpp_tests(test_intervals)

endif ()
