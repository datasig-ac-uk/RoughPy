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

#ifndef ROUGHPY_STREAMS_CATEGORICAL_CHANNEL_H
#define ROUGHPY_STREAMS_CATEGORICAL_CHANNEL_H

#include "stream_channel.h"

namespace rpy {
namespace streams {

class ROUGHPY_STREAMS_EXPORT CategoricalChannel : public StreamChannel
{
    std::vector<string> m_variants;

public:

    CategoricalChannel() : StreamChannel(ChannelType::Categorical, nullptr)
    {}

    dimn_t num_variants() const override;
    string label_suffix(dimn_t variant_no) const override;
    dimn_t variant_id_of_label(string_view label) const override;
    const std::vector<string>& get_variants() const override;

    StreamChannel& add_variant(string variant_label) override;
    StreamChannel& insert_variant(string variant_label) override;

    RPY_SERIAL_SERIALIZE_FN();
};

#ifdef RPY_COMPILING_STREAMS
RPY_SERIAL_EXTERN_SERIALIZE_CLS_BUILD(CategoricalChannel)
#else
RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMP(CategoricalChannel)
#endif

RPY_SERIAL_SERIALIZE_FN_IMPL(CategoricalChannel) {
    RPY_SERIAL_SERIALIZE_BASE(StreamChannel);
    RPY_SERIAL_SERIALIZE_NVP("variants", m_variants);
}



}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::streams::CategoricalChannel,
                            rpy::serial::specialization::member_serialize)


RPY_SERIAL_FORCE_DYNAMIC_INIT(categorical_channel);

#endif// ROUGHPY_STREAMS_CATEGORICAL_CHANNEL_H
