


add_library(RoughPy_Scalar_Type_rational STATIC
    rational.h
    rational_type.cpp
    rational_type.h
    arbitrary_precision_rational_type.cpp
    arbitrary_precision_rational_type.h
    arbitrary_precision_rational.h)

target_link_libraries(RoughPy_Scalar_Type_rational
    PUBLIC
    RoughPy::Core
    RoughPy::Platform
    PRIVATE
    Boost::boost
)

target_include_directories(RoughPy_Scalar_Type_rational PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

set_target_properties(RoughPy_Scalar_Type_rational
    PROPERTIES POSITION_INDEPENDENT_CODE ON)
