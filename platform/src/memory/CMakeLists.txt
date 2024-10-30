

set(INCLUDE_ROOT "${RoughPy_Platform_SOURCE_DIR}/include/roughpy")

set(CONTAINERS_ROOT "${INCLUDE_ROOT}/containers")

target_sources(RoughPy_Platform PRIVATE
    ${INCLUDE_ROOT}/platform/memory.h
    ${CONTAINERS_ROOT}/deque.h
    ${CONTAINERS_ROOT}/hash_map.h
    ${CONTAINERS_ROOT}/flat_map.h
    ${CONTAINERS_ROOT}/flat_multimap.h
    ${CONTAINERS_ROOT}/flat_set.h
    ${CONTAINERS_ROOT}/hash_set.h
    ${CONTAINERS_ROOT}/list.h
    ${CONTAINERS_ROOT}/map.h
    ${CONTAINERS_ROOT}/set.h
    ${CONTAINERS_ROOT}/small_flat_map.h
    ${CONTAINERS_ROOT}/small_vector.h
    ${CONTAINERS_ROOT}/stable_vector.h
    ${CONTAINERS_ROOT}/vector.h
    aligned_alloc.cpp
    small_object_base.cpp
    AlignedMemory.cpp
    AlignedMemory.h
    small_object_memory.cpp
    small_object_alloc.cpp
    small_object_memory.h
)

target_include_directories(RoughPy_Platform PRIVATE
        "${RoughPy_Platform_SOURCE_DIR}/include"
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

    if (WIN32)
        add_custom_command(TARGET RoughPy_Platform_test_memory POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:RoughPy_Platform_test_memory > $<TARGET_RUNTIME_DLLS:RoughPy_Platform_test_memory >
                COMMAND_EXPAND_LISTS)
    endif ()

    gtest_discover_tests(RoughPy_Platform_test_memory
            EXTRA_ARGS --rerun-failed --output-on-failure
            DISCOVERY_MODE POST_BUILD
            TEST_LIST memory_tests
    )



endif ()