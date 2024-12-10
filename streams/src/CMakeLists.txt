


add_subdirectory(channels)
add_subdirectory(external_data_sources)



if (ROUGHPY_BUILD_TESTS)

    add_executable(test_streams
            test_lie_increment_stream.cpp
            test_brownian_stream.cpp
            test_schema.cpp
            test_tensor_valued_stream.cpp
    )

    target_link_libraries(test_streams PRIVATE RoughPy::Streams GTest::gtest)

    setup_roughpy_cpp_tests(test_streams)

endif()