


target_sources(RoughPy_Streams PRIVATE
        categorical_channel.cpp
        stream_channel.cpp
        increment_channel.cpp
        value_channel.cpp
        lie_channel.cpp
        lead_laggable_channel.cpp

        ${ROUGHPY_STREAMS_INCLUDE_DIR}/channels/stream_channel.h
        ${ROUGHPY_STREAMS_INCLUDE_DIR}/channels/increment_channel.h
        ${ROUGHPY_STREAMS_INCLUDE_DIR}/channels/value_channel.h
        ${ROUGHPY_STREAMS_INCLUDE_DIR}/channels/categorical_channel.h
        ${ROUGHPY_STREAMS_INCLUDE_DIR}/channels/lie_channel.h
        ${ROUGHPY_STREAMS_INCLUDE_DIR}/channels/lead_laggable_channel.h
)

