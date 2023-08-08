//
// Created by sam on 07/08/23.
//

#ifndef ROUGHPY_STREAMS_INCREMENT_CHANNEL_H
#define ROUGHPY_STREAMS_INCREMENT_CHANNEL_H

#include "lead_laggable_channel.h"

namespace rpy { namespace streams {



class RPY_EXPORT IncrementChannel : public LeadLaggableChannel
{
public:

    IncrementChannel() : LeadLaggableChannel(ChannelType::Increment, nullptr)
    {}


    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(IncrementChannel) {
    RPY_SERIAL_SERIALIZE_BASE(LeadLaggableChannel);
}

}}

RPY_SERIAL_SPECIALIZE_TYPES(rpy::streams::IncrementChannel,
                            rpy::serial::specialization::member_serialize)

RPY_SERIAL_REGISTER_CLASS(rpy::streams::IncrementChannel)

#endif// ROUGHPY_STREAMS_INCREMENT_CHANNEL_H
