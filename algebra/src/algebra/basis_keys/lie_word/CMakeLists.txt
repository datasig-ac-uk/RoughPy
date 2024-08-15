

add_library(RoughPy_lie_word STATIC
        lie_word.cpp
        lie_word.h
        lie_word_type.cpp
        lie_word_type.h
)

set_target_properties(RoughPy_lie_word PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(RoughPy_lie_word PUBLIC RoughPy::Core RoughPy::Platform)


if (ROUGHPY_BUILD_TESTS)
    add_executable(test_lie_word test_lie_word.cpp)

    target_link_libraries(test_lie_word PRIVATE
            RoughPy_lie_word
            GTest::gtest_main)


    gtest_discover_tests(test_lie_word)
endif ()
