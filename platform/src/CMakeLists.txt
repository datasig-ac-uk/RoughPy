
add_subdirectory(generics)
add_subdirectory(devices)

if(ROUGHPY_EXPERIMENTAL)
    add_subdirectory(device)
endif()


if (ROUGHPY_BUILD_TESTS)

    add_executable(test_platform
            test_alloc.cpp
            test_polymorphic_ref_counter.cpp
    )

    target_link_libraries(test_platform PRIVATE RoughPy_Platform GTest::gtest)

    setup_roughpy_cpp_tests(test_platform)

endif()