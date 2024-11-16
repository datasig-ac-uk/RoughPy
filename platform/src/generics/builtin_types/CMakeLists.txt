



add_library(RoughPy_Generics_BuiltinType STATIC
        builtin_type.h
        builtin_type_methods.h
        double_type.cpp
        double_type.h
)

target_include_directories(RoughPy_Generics_BuiltinType PRIVATE
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/../../..
)

target_link_libraries(RoughPy_Generics_BuiltinType PRIVATE
        RoughPy::Core
)


