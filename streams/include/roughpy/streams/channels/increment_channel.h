// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

RPY_SERIAL_EXTERN_SERIALIZE_CLS(IncrementChannel)

RPY_SERIAL_SERIALIZE_FN_IMPL(IncrementChannel) {
    RPY_SERIAL_SERIALIZE_BASE(LeadLaggableChannel);
}

}}

RPY_SERIAL_SPECIALIZE_TYPES(rpy::streams::IncrementChannel,
                            rpy::serial::specialization::member_serialize)

RPY_SERIAL_REGISTER_CLASS(rpy::streams::IncrementChannel)

#endif// ROUGHPY_STREAMS_INCREMENT_CHANNEL_H
