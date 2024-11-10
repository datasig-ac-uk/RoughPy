


target_sources(RoughPy_Streams PRIVATE
        categorical_channel.cpp
        stream_channel.cpp
        increment_channel.cpp
        value_channel.cpp
        lie_channel.cpp
        lead_laggable_channel.cpp

        ${ROUGHPY_STREAMS_INCLUDE}/channels/stream_channel.h
        ${ROUGHPY_STREAMS_INCLUDE}/channels/increment_channel.h
        ${ROUGHPY_STREAMS_INCLUDE}/channels/value_channel.h
        ${ROUGHPY_STREAMS_INCLUDE}/channels/categorical_channel.h
        ${ROUGHPY_STREAMS_INCLUDE}/channels/lie_channel.h
        ${ROUGHPY_STREAMS_INCLUDE}/channels/lead_laggable_channel.h
)

