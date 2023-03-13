cmake_minimum_required(VERSION 3.19)



project(RoughPy_PyModule
        VERSION 0.0.1
        LANGUAGES CXX
        )

option(ROUGHPY_LINK_NUMPY "Link with Numpy library for array handling" ON)


if(ROUGHPY_LINK_NUMPY)
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
else()
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
endif()


if(NOT TARGET pybind11::headers)
    find_package(Pybind11 CONFIG REQUIRED)
endif()



Python_add_library(RoughPy_PyModule MODULE WITH_SOABI)
set_target_properties(RoughPy_PyModule PROPERTIES
        LIBRARY_OUTPUT_NAME _roughpy
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})


target_sources(RoughPy_PyModule PRIVATE
        src/roughpy_module.cpp
        src/recombine.cpp
        src/recombine.h
        src/algebra/basis.cpp
        src/algebra/basis.h
        src/algebra/context.cpp
        src/algebra/context.h
        src/algebra/free_tensor.h
        src/algebra/free_tensor.cpp
        src/algebra/lie.cpp
        src/algebra/lie.h
        src/algebra/lie_key.cpp
        src/algebra/lie_key.h
        src/algebra/lie_key_iterator.cpp
        src/algebra/lie_key_iterator.h
        src/algebra/lie_letter.cpp
        src/algebra/lie_letter.h
        src/algebra/shuffle_tensor.h
        src/algebra/shuffle_tensor.cpp
        src/algebra/tensor_key.cpp
        src/algebra/tensor_key.h
        src/algebra/tensor_key_iterator.cpp
        src/algebra/tensor_key_iterator.h
        src/args/kwargs_to_path_metadata.h
        src/args/kwargs_to_path_metadata.cpp
        src/args/kwargs_to_vector_construction.cpp
        src/args/kwargs_to_vector_construction.h
        src/args/numpy.cpp
        src/args/numpy.h
        src/intervals/dyadic.h
        src/intervals/dyadic.cpp
        src/intervals/dyadic_interval.cpp
        src/intervals/dyadic_interval.h
        src/intervals/interval.h
        src/intervals/interval.cpp
        src/intervals/intervals.cpp
        src/intervals/intervals.h
        src/intervals/real_interval.cpp
        src/intervals/real_interval.h
        src/intervals/segmentation.cpp
        src/intervals/segmentation.h
        src/scalars/scalar_type.h
        src/scalars/scalar_type.cpp
        src/scalars/scalars.cpp
        src/scalars/scalars.h
        src/streams/function_path.cpp
        src/streams/function_path.h
        src/streams/lie_increment_stream.h
        src/streams/lie_increment_stream.cpp
        src/streams/stream.cpp
        src/streams/stream.h
        src/streams/streams.cpp
        src/streams/streams.h
        src/streams/tick_stream.h
        src/streams/tick_stream.cpp
        )


target_link_libraries(RoughPy_PyModule PRIVATE
        RoughPy::Scalars
        RoughPy::Intervals
        RoughPy::Algebra
        RoughPy::Streams
        pybind11::headers
        )

if(ROUGHPY_LINK_NUMPY)
    target_compile_definitions(RoughPy_PyModule PRIVATE
        ROUGHPY_WITH_NUMPY)
endif()
