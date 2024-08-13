
add_library(RoughPy_tensor_word STATIC
        tensor_word.cpp
        tensor_word.h
        tensor_word_type.cpp
        tensor_word_type.h
)

set_target_properties(RoughPy_tensor_word
        PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_link_libraries(RoughPy_tensor_word PUBLIC
        RoughPy::Core
        RoughPy::Platform
)
