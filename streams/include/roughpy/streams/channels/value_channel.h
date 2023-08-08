//
// Created by sam on 07/08/23.
//

#ifndef ROUGHPY_STREAMS_VALUE_CHANNEL_H
#define ROUGHPY_STREAMS_VALUE_CHANNEL_H

#include "lead_laggable_channel.h"

namespace rpy {
namespace streams {

class RPY_EXPORT ValueChannel : public LeadLaggableChannel
{

public:
    ValueChannel() : LeadLaggableChannel(ChannelType::Value, nullptr) {}

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(ValueChannel)
{
    RPY_SERIAL_SERIALIZE_BASE(LeadLaggableChannel);
}

}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::streams::ValueChannel,
        rpy::serial::specialization::member_serialize
)

RPY_SERIAL_REGISTER_CLASS(rpy::streams::ValueChannel)

#endif// ROUGHPY_STREAMS_VALUE_CHANNEL_H
