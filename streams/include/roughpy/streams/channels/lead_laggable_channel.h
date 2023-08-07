//
// Created by sam on 07/08/23.
//

#ifndef ROUGHPY_STREAMS_LEAD_LAGGABLE_CHANNEL_H
#define ROUGHPY_STREAMS_LEAD_LAGGABLE_CHANNEL_H

#include "stream_channel.h"

namespace rpy {
namespace streams {

class RPY_EXPORT LeadLaggableChannel : public StreamChannel
{
    bool m_use_leadlag = false;

protected:

    using StreamChannel::StreamChannel;

public:
    dimn_t num_variants() const override;
    string label_suffix(dimn_t variant_no) const override;
    dimn_t variant_id_of_label(string_view label) const override;
    const std::vector<string>& get_variants() const override;
    void set_lead_lag(bool new_value) override;
    bool is_lead_lag() const override;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(LeadLaggableChannel)
{
    RPY_SERIAL_SERIALIZE_BASE(StreamChannel);
    RPY_SERIAL_SERIALIZE_NVP("use_leadlag", m_use_leadlag);
}

}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::streams::LeadLaggableChannel,
        rpy::serial::specialization::member_serialize
)


#endif// ROUGHPY_STREAMS_LEAD_LAGGABLE_CHANNEL_H
