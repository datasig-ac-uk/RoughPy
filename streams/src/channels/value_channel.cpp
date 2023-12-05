//
// Created by sam on 07/08/23.
//

#include <roughpy/streams/channels/value_channel.h>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::ValueChannel
#include <roughpy/platform/serialization_instantiations.inl>

RPY_SERIAL_REGISTER_CLASS(rpy::streams::ValueChannel)

RPY_SERIAL_DYNAMIC_INIT(value_channel)
