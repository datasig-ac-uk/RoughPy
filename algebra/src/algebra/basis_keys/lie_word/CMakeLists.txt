
add_library(RoughPy_lie_word STATIC)

set_target_properties(RoughPy_lie_word PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(RoughPy_lie_word PUBLIC RoughPy::Core RoughPy::Platform)
