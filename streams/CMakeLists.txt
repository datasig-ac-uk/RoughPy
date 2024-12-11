

setup_roughpy_component(Streams)


add_library(RoughPy_Streams SHARED
        src/stream_base.cpp src/stream.cpp
        src/lie_increment_stream.cpp
        src/dynamically_constructed_stream.cpp
        src/dyadic_caching_layer.cpp
        src/piecewise_abelian_stream.cpp
        src/tick_stream.cpp
        src/brownian_stream.cpp
        src/external_data_stream.cpp
        src/schema.cpp
        src/parametrization.cpp
        src/stream_construction_helper.cpp
        include/roughpy/streams/arrival_stream.h
        include/roughpy/streams/dyadic_caching_layer.h
        include/roughpy/streams/dynamically_constructed_stream.h
        include/roughpy/streams/lie_increment_stream.h
        include/roughpy/streams/stream.h
        include/roughpy/streams/stream_base.h
        include/roughpy/streams/tick_stream.h
        include/roughpy/streams/brownian_stream.h
        include/roughpy/streams/external_data_stream.h
        include/roughpy/streams/piecewise_abelian_stream.h
        include/roughpy/streams/schema.h
        include/roughpy/streams/parametrization.h
        include/roughpy/streams/stream_construction_helper.h
        include/roughpy/streams/channels.h
        include/roughpy/streams/value_stream.h
        src/arrival_stream.cpp
        include/roughpy/streams/restriction_stream.h
        src/restriction_stream.cpp
        src/tensor_valued_stream.cpp
)
add_library(RoughPy::Streams ALIAS RoughPy_Streams)


target_include_directories(RoughPy_Streams PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/include/roughpy/streams
)

target_link_libraries(RoughPy_Streams PUBLIC
        RoughPy::Algebra
        RoughPy::Intervals
        Boost::url
        PRIVATE
            RoughPy::PrecompiledHeaders
)

set_target_properties(RoughPy_Streams PROPERTIES ROUGHPY_COMPONENT Streams)


# TODO: This should be removed, it's not sensible
target_compile_definitions(RoughPy_Streams PRIVATE "RPY_COMPILING_STREAMS")

set_library_version_properties(RoughPy_Streams)



generate_export_header(RoughPy_Streams)

add_subdirectory(src)


