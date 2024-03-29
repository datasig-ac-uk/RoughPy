cmake_minimum_required(VERSION 3.21)


project(RoughPy_PyModule
        VERSION ${RoughPy_VERSION}
        LANGUAGES CXX
)
message(STATUS "Adding RoughPy::PyModule version ${RoughPy_VERSION}")


if (NOT TARGET pybind11::headers)
    find_package(Pybind11 CONFIG REQUIRED)
endif ()


Python_add_library(RoughPy_PyModule MODULE WITH_SOABI)
set_target_properties(RoughPy_PyModule PROPERTIES
        LIBRARY_OUTPUT_NAME _roughpy
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
)

target_compile_definitions(RoughPy_PyModule PRIVATE
        RPY_BUILDING_LIBRARY=1
        BOOST_UUID_FORCE_AUTO_LINK=1
)

if (ROUGHPY_BUILD_PYMODULE_INPLACE)
    if (MSVC)
        add_custom_command(TARGET RoughPy_PyModule POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy -t ${CMAKE_CURRENT_LIST_DIR} $<TARGET_FILE:RoughPy_PyModule>
        )
        add_custom_command(TARGET RoughPy_PyModule POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy -t ${CMAKE_CURRENT_LIST_DIR} $<TARGET_RUNTIME_DLLS:RoughPy_PyModule>
            COMMAND_EXPAND_LISTS
        )
    else()
        set_target_properties(RoughPy_PyModule PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    endif ()
endif ()


#set_target_properties(RoughPy_PyModule PROPERTIES
#    INSTALL_RPATH $ORIGIN)
if (WIN32)

elseif (APPLE)
    #    set_target_properties(RoughPy_PyModule PROPERTIES
    #            INSTALL_RPATH "@loader_path;@loader_path/../../..")
else ()
    set_target_properties(RoughPy_PyModule PROPERTIES
            INSTALL_RPATH $ORIGIN)
endif ()

target_sources(RoughPy_PyModule PRIVATE
        src/roughpy_module.cpp
        src/roughpy_module.h
        #        src/recombine.cpp
        #        src/recombine.h
        src/algebra/algebra.h
        src/algebra/algebra.cpp
        src/algebra/algebra_iterator.cpp
        src/algebra/algebra_iterator.h
        src/algebra/basis.cpp
        src/algebra/basis.h
        src/algebra/context.cpp
        src/algebra/context.h
        src/algebra/free_multiply_funcs.cpp
        src/algebra/free_multiply_funcs.h
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
        src/algebra/setup_algebra_type.h
        src/algebra/shuffle_tensor.h
        src/algebra/shuffle_tensor.cpp
        src/algebra/tensor_key.cpp
        src/algebra/tensor_key.h
        src/algebra/tensor_key_iterator.cpp
        src/algebra/tensor_key_iterator.h
        src/args/buffer_info.cpp
        src/args/buffer_info.h
        src/args/check_for_excess_args.cpp
        src/args/convert_timestamp.cpp
        src/args/convert_timestamp.h
        src/args/dlpack_helpers.cpp
        src/args/dlpack_helpers.h
        src/args/kwargs_to_path_metadata.h
        src/args/kwargs_to_path_metadata.cpp
        src/args/kwargs_to_vector_construction.cpp
        src/args/kwargs_to_vector_construction.h
        src/args/numpy.cpp
        src/args/numpy.h
        src/args/parse_algebra_configuration.cpp
        src/args/parse_algebra_configuration.h
        src/args/parse_data_argument.cpp
        src/args/parse_data_argument.h
        src/args/parse_schema.cpp
        src/args/parse_schema.h
        src/scalars/pytype_conversion.cpp
        src/scalars/pytype_conversion.h
        src/args/strided_copy.cpp
        src/args/strided_copy.h
        src/intervals/date_time_interval.cpp
        src/intervals/date_time_interval.h
        src/intervals/dyadic.h
        src/intervals/dyadic.cpp
        src/intervals/dyadic_interval.cpp
        src/intervals/dyadic_interval.h
        src/intervals/interval.h
        src/intervals/interval.cpp
        src/intervals/intervals.cpp
        src/intervals/intervals.h
        src/intervals/partition.cpp
        src/intervals/partition.h
        src/intervals/real_interval.cpp
        src/intervals/real_interval.h
        src/intervals/segmentation.cpp
        src/intervals/segmentation.h
        src/scalars/r_py_polynomial.cpp
        src/scalars/r_py_polynomial.h
        #        src/scalars/scalar.h
        #        src/scalars/scalar.cpp
        src/scalars/scalar_type.h
        src/scalars/scalar_type.cpp
        src/scalars/scalars.cpp
        src/scalars/scalars.h
        src/streams/BaseStream.cpp
        src/streams/BaseStream.h
        src/streams/brownian_stream.cpp
        src/streams/brownian_stream.h
        src/streams/externally_sourced_stream.cpp
        src/streams/externally_sourced_stream.h
        src/streams/function_stream.cpp
        src/streams/function_stream.h
        src/streams/lie_increment_stream.h
        src/streams/lie_increment_stream.cpp
        src/streams/piecewise_abelian_stream.cpp
        src/streams/piecewise_abelian_stream.h
        src/streams/r_py_tick_construction_helper.cpp
        src/streams/r_py_tick_construction_helper.h
        src/streams/py_parametrization.cpp
        src/streams/py_parametrization.h
        src/streams/schema.cpp
        src/streams/schema.h
        src/streams/schema_finalization.cpp
        src/streams/schema_finalization.h
        src/streams/stream.cpp
        src/streams/stream.h
        src/streams/streams.cpp
        src/streams/streams.h
        src/streams/tick_stream.h
        src/streams/tick_stream.cpp
)

target_include_directories(RoughPy_PyModule PRIVATE
        src/
)

target_link_libraries(RoughPy_PyModule PRIVATE
        pybind11::headers
        Boost::boost
        RoughPy::Core
        RoughPy::Platform
        RoughPy::Scalars
        RoughPy::Intervals
        RoughPy::Algebra
        RoughPy::Streams
        #        $<LINK_LIBRARY:WHOLE_ARCHIVE,RoughPy::Streams>
        #        recombine::recombine
)


target_precompile_headers(RoughPy_PyModule PRIVATE
        <roughpy/platform/devices.h>
        <roughpy/platform/serialization.h>
        <roughpy/platform/filesystem.h>
        <roughpy/scalars/key_scalar_array.h>
        <roughpy/algebra/context.h>
        <pybind11/pybind11.h>
        <pybind11/functional.h>
        <pybind11/stl.h>
)

set(RPY_TARGETS RoughPy::Platform
        RoughPy::Scalars
        RoughPy::Algebra
        RoughPy::Intervals
        RoughPy::Streams)


if (ROUGHPY_LINK_NUMPY)
    target_link_libraries(RoughPy_PyModule PRIVATE Python::NumPy)
    target_compile_definitions(RoughPy_PyModule PRIVATE
            ROUGHPY_WITH_NUMPY)
    target_precompile_headers(RoughPy_PyModule PRIVATE <pybind11/numpy.h>)
endif ()


get_target_property(RoughPy_Core_VERSION RoughPy::Core VERSION)
get_target_property(RoughPy_Platform_VERSION RoughPy::Platform VERSION)
get_target_property(RoughPy_Scalars_VERSION RoughPy::Scalars VERSION)
get_target_property(RoughPy_Intervals_VERSION RoughPy::Intervals VERSION)
get_target_property(RoughPy_Algebra_VERSION RoughPy::Algebra VERSION)
get_target_property(RoughPy_Streams_VERSION RoughPy::Streams VERSION)


configure_file(${CMAKE_SOURCE_DIR}/cmake/version.py.in ${CMAKE_CURRENT_LIST_DIR}/version.py @ONLY)
