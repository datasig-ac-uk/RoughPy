

set(INCLUDE_ROOT "${RoughPy_Platform_SOURCE_DIR}/include/roughpy/")

target_sources(RoughPy_Platform PRIVATE
    ${INCLUDE_ROOT}/platform/memory.h
    aligned_alloc.cpp
    small_object_base.cpp
    AlignedMemory.cpp
    AlignedMemory.h
    small_object_memory.cpp
    small_object_alloc.cpp
    small_object_memory.h
)


if (ROUGHPY_BUILD_TESTS)

    add_executable(RoughPy_Platform_test_memory
        test_aligned_allocator.cpp
        test_small_object_allocator.cpp
        test_ref_count_pointer.cpp
    )

    target_link_libraries(RoughPy_Platform_test_memory PRIVATE
        GTest::gtest_main
        RoughPy_Platform
    )

    gtest_discover_tests(RoughPy_Platform_test_memory)
endif ()