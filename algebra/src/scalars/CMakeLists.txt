
add_subdirectory(scalar_types)


add_library(RoughPy_scalars STATIC
        scalar_array.cpp
        scalar_stream.cpp
        scalar_vector.cpp
        vector_data.cpp
        scalar_vector_iterator.cpp
        vector_operation.cpp
        vector_data.h
)
target_link_libraries(RoughPy_scalars PUBLIC RoughPy::Core RoughPy::Platform
)
target_include_directories(RoughPy_scalars PRIVATE ${ROUGHPY_SCALARS_PRIVATE_HEADERS})
set_target_properties(RoughPy_scalars PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_sources(RoughPy_scalars PRIVATE
        ${ROUGHPY_SCALARS_PRIVATE_HEADERS}/scalar_array.h
        ${ROUGHPY_SCALARS_PRIVATE_HEADERS}/scalar_stream.h
        ${ROUGHPY_SCALARS_PRIVATE_HEADERS}/scalar_vector.h
        ${ROUGHPY_SCALARS_PRIVATE_HEADERS}/scalars_fwd.h
)
