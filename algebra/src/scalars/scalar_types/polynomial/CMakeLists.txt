



add_library(RoughPy_monomial INTERFACE)

target_link_libraries(RoughPy_monomial INTERFACE RoughPy::Core RoughPy::Platform)
target_include_directories(RoughPy_monomial INTERFACE ${CMAKE_CURRENT_LIST_DIR})

add_library(RoughPy_Scalar_Type_polynomial STATIC
    poly_rational.h
    poly_rational_type.cpp
    poly_rational_type.h)


target_link_libraries(RoughPy_Scalar_Type_polynomial
    PUBLIC
    RoughPy::Core RoughPy::Platform RoughPy_Scalar_Type_rational RoughPy_monomial
)

target_include_directories(RoughPy_Scalar_Type_polynomial PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR})

set_target_properties(RoughPy_Scalar_Type_polynomial
    PROPERTIES POSITION_INDEPENDENT_CODE ON)