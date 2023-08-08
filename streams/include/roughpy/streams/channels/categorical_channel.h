//
// Created by sam on 07/08/23.
//

#ifndef ROUGHPY_STREAMS_CATEGORICAL_CHANNEL_H
#define ROUGHPY_STREAMS_CATEGORICAL_CHANNEL_H

#include "stream_channel.h"

namespace rpy {
namespace streams {

class RPY_EXPORT CategoricalChannel : public StreamChannel
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


RPY_SERIAL_SERIALIZE_FN_IMPL(CategoricalChannel) {
    RPY_SERIAL_SERIALIZE_BASE(StreamChannel);
    RPY_SERIAL_SERIALIZE_NVP("variants", m_variants);
}



}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::streams::CategoricalChannel,
                            rpy::serial::specialization::member_serialize)


RPY_SERIAL_REGISTER_CLASS(rpy::streams::CategoricalChannel)

#endif// ROUGHPY_STREAMS_CATEGORICAL_CHANNEL_H

