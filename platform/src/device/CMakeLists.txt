
set(_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/device")

target_sources(RoughPy_Platform PUBLIC
        ${_header_dir}/device_handle.h
        ${_header_dir}/event.h
        ${_header_dir}/host_address_memory.h
        ${_header_dir}/memory.h
        ${_header_dir}/queue.h

        PRIVATE
        device_handle.cpp
        event.cpp
        host_address_memory.cpp
        host_device.cpp
        host_device.h
        memory.cpp
        queue.cpp
)


if (ROUGHPY_BUILD_TESTS)

    add_executable(test_device
        test_host_address_memory.cpp
    )

    target_link_libraries(test_device PRIVATE
            RoughPy::Platform
            GTest::gtest
            GTest::gmock
    )

    setup_roughpy_cpp_tests(test_device)

endif()