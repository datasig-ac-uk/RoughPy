



target_sources(RoughPy_Algebra PRIVATE
    tensor_basis.h
    tensor_basis.cpp
    to_letter_iterator.cpp
    to_letter_iterator.h
)


if (ROUGHPY_BUILD_TESTS)

    add_executable(test_tensor_basis test_tensor_basis.cpp)

    target_link_libraries(test_tensor_basis PRIVATE RoughPy_Algebra GTest::gtest_main)

endif ()
