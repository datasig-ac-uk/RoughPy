

add_library(RoughPy_index_key STATIC index_key_type.cpp index_key_type.h)


set_target_properties(RoughPy_index_key PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(RoughPy_index_key PUBLIC RoughPy::Core RoughPy::Platform)

target_include_directories(RoughPy_index_key PUBLIC ${CMAKE_CURRENT_LIST_DIR})


if (ROUGHPY_BUILD_TESTS)


endif ()
