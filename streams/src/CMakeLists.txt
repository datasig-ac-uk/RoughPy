


add_subdirectory(channels)
add_subdirectory(external_data_sources)



if (ROUGHPY_BUILD_TESTS)

    add_executable(test_streams
            test_lie_increment_stream.cpp
#            test_brownian_stream.cpp // TODO: There is a bug in the refine_accuracy function.
            test_schema.cpp
    )

    target_link_libraries(test_streams PRIVATE RoughPy::Streams GTest::gtest)

    setup_roughpy_cpp_tests(test_streams)

endif()