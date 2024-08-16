
add_library(RoughPy_tensor_word STATIC
        tensor_word.cpp
        tensor_word.h
        tensor_word_type.cpp
        tensor_word_type.h
)

set_target_properties(RoughPy_tensor_word
        PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_link_libraries(RoughPy_tensor_word PUBLIC
        RoughPy::Core
        RoughPy::Platform
)

target_include_directories(RoughPy_tensor_word PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if (ROUGHPY_BUILD_TESTS)
    add_executable(test_tensor_word
            test_tensor_word.cpp)

    target_link_libraries(test_tensor_word PRIVATE
            RoughPy_tensor_word
            GTest::gtest_main
    )

    gtest_discover_tests(test_tensor_word)
endif ()
