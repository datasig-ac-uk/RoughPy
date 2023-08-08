//
// Created by sam on 07/08/23.
//

#ifndef ROUGHPY_STREAMS_LIE_CHANNEL_H
#define ROUGHPY_STREAMS_LIE_CHANNEL_H

#include "stream_channel.h"

namespace rpy {
namespace streams {

class RPY_EXPORT LieChannel : public StreamChannel
{
public:
    LieChannel() : StreamChannel(ChannelType::Lie, nullptr) {}

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(LieChannel)
{
    RPY_SERIAL_SERIALIZE_BASE(StreamChannel);
}

}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::streams::LieChannel, rpy::serial::specialization::member_serialize
)

RPY_SERIAL_REGISTER_CLASS(rpy::streams::LieChannel)
#endif// ROUGHPY_STREAMS_LIE_CHANNEL_H
