//
// Created by sam on 07/08/23.
//

#include <roughpy/streams/channels/value_channel.h>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::ValueChannel
#define RPY_EXPORT_MACRO ROUGHPY_STREAMS_EXPORT
#define RPY_SERIAL_DO_REGISTER
#include <roughpy/platform/serialization_instantiations.inl>


RPY_SERIAL_DYNAMIC_INIT(value_channel)
