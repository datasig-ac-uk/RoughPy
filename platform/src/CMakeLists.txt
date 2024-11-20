
if(ROUGHPY_EXPERIMENTAL)
add_subdirectory(generics)
endif()

add_subdirectory(devices)

if(ROUGHPY_EXPERIMENTAL)
    add_subdirectory(new_devices)
endif()


if (ROUGHPY_BUILD_TESTS)

    add_executable(test_platform
            test_alloc.cpp
    )

    target_link_libraries(test_platform PRIVATE RoughPy_Platform GTest::gtest)

    setup_roughpy_cpp_tests(test_platform)

endif()