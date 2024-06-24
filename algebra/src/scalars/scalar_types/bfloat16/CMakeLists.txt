
message(STATUS "Adding blfoat16 type")
add_library(RoughPy_Scalar_Type_bfloat16 STATIC
    bfloat16.h
    bfloat_16_type.cpp
    bfloat_16_type.h)

target_link_libraries(RoughPy_Scalar_Type_bfloat16 PUBLIC
    PUBLIC
    RoughPy::Core
    RoughPy::Platform
    PRIVATE
    Eigen3::Eigen
)

set_target_properties(RoughPy_Scalar_Type_bfloat16
    PROPERTIES POSITION_INDEPENDENT_CODE ON)