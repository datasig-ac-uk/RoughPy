cmake_minimum_required(VERSION 3.21)

project(RoughPy_Streams
        VERSION 0.0.1
        LANGUAGES CXX)


include(GNUInstallDirs)
include(GenerateExportHeader)


add_roughpy_component(Streams
    SOURCES
        src/stream_base.cpp
        src/stream.cpp
        src/lie_increment_stream.cpp
        src/dynamically_constructed_stream.cpp
        src/dyadic_caching_layer.cpp
        src/piecewise_abelian_stream.cpp
        src/tick_stream.cpp
        src/brownian_stream.cpp
        src/external_data_stream.cpp
        src/external_data_sources/csv_data_source.cpp
        src/external_data_sources/csv_data_source.h
        src/external_data_sources/sound_file_data_source.cpp
        src/external_data_sources/sound_file_data_source.h
        src/schema.cpp
        src/parametrization.cpp
        src/stream_construction_helper.cpp
        src/channels/categorical_channel.cpp
        src/channels/stream_channel.cpp
        src/channels/increment_channel.cpp
        src/channels/value_channel.cpp
        src/channels/lie_channel.cpp
        src/channels/lead_laggable_channel.cpp
    PUBLIC_HEADERS
        include/roughpy/streams/channels/stream_channel.h
        include/roughpy/streams/channels/increment_channel.h
        include/roughpy/streams/channels/value_channel.h
        include/roughpy/streams/channels/categorical_channel.h
        include/roughpy/streams/channels/lie_channel.h
        include/roughpy/streams/channels/lead_laggable_channel.h
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
    DEPENDENCIES PRIVATE
        SndFile::sndfile
        PUBLIC
                Boost::boost
                Boost::url
    DEFINITIONS
        BOOST_UUID_FORCE_AUTO_LINK=1
    NEEDS
        RoughPy::Core
        RoughPy::Platform
        RoughPy::Intervals
        RoughPy::Scalars
        RoughPy::Algebra
        )

target_precompile_headers(RoughPy_Streams PRIVATE
        <roughpy/scalars/key_scalar_array.h>
        <roughpy/algebra/context.h>
        <roughpy/algebra/free_tensor.h>
        <roughpy/algebra/lie.h>
)


add_roughpy_test(Streams
        SRC
        src/test_lie_increment_stream.cpp
        src/test_brownian_stream.cpp
        src/test_schema.cpp
        NEEDS
        RoughPy::Streams
        )
