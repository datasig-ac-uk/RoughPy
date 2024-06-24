


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