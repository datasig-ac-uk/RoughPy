


add_library(RoughPy_hall_basis STATIC
        hall_basis.cpp
        hall_basis.h)

set_target_properties(RoughPy_hall_basis PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(RoughPy_hall_basis PUBLIC
        RoughPy::Core
        RoughPy::Platform
        RoughPy_Algebra_headers
        RoughPy_lie_word
        RoughPy_index_key)

target_include_directories(RoughPy_hall_basis PUBLIC ${CMAKE_CURRENT_LIST_DIR})
