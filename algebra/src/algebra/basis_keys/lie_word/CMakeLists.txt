

add_library(RoughPy_lie_word STATIC
        lie_word.cpp
        lie_word.h
        lie_word_type.cpp
        lie_word_type.h
)

set_target_properties(RoughPy_lie_word PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(RoughPy_lie_word PUBLIC RoughPy::Core RoughPy::Platform)
