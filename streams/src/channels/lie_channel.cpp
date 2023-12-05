//
// Created by sam on 07/08/23.
//

#include <roughpy/streams/channels/lie_channel.h>






#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::LieChannel
#include <roughpy/platform/serialization_instantiations.inl>

RPY_SERIAL_REGISTER_CLASS(rpy::streams::LieChannel)


RPY_SERIAL_DYNAMIC_INIT(lie_channel)
