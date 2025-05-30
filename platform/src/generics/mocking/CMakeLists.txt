set(_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/generics/mocking")

add_library(RoughPy_Platform_Mocking
    STATIC
        ${_header_dir}/mock_type.h
        mock_type.cpp
)

target_link_libraries(RoughPy_Platform_Mocking
    PRIVATE
        RoughPy::Platform
)
