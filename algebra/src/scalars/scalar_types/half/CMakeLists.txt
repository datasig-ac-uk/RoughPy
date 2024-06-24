

add_library(RoughPy_Scalar_Type_half STATIC
    half.h
    half_type.cpp
    half_type.h
)


target_link_libraries(RoughPy_Scalar_Type_half
    PUBLIC
        RoughPy::Core
        RoughPy::Platform
    PRIVATE Eigen3::Eigen)

set_target_properties(RoughPy_Scalar_Type_half
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

