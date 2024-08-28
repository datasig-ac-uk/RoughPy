


#add_library(RoughPy_hall_basis STATIC
#    hall_basis.cpp
#    hall_basis.h
#    hall_set_size.cpp
#    hall_set_size.h)

target_sources(RoughPy_Algebra PRIVATE hall_basis.cpp ${ROUGHPY_ALGEBRA_PRIVATE_HEADERS}/bases/hall_basis.h hall_set_size.cpp hall_set_size.h)


#set_target_properties(RoughPy_hall_basis PROPERTIES POSITION_INDEPENDENT_CODE ON)
#
#target_link_libraries(RoughPy_hall_basis PUBLIC
#    RoughPy::Core
#    RoughPy::Platform
#    RoughPy_Algebra_headers
#    RoughPy_lie_word
#    RoughPy_index_key)

#target_include_directories(RoughPy_hall_basis PUBLIC ${CMAKE_CURRENT_LIST_DIR})


if (ROUGHPY_BUILD_TESTS)

    add_executable(test_hall_basis test_hall_basis.cpp)

    target_link_libraries(test_hall_basis PRIVATE RoughPy_Algebra GTest::gtest_main)

endif ()
