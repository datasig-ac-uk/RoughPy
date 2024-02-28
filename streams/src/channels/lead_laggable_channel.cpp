//
// Created by sam on 07/08/23.
//

#include <roughpy/streams/channels/lead_laggable_channel.h>

using namespace rpy;
using namespace rpy::streams;

dimn_t LeadLaggableChannel::num_variants() const
{
    return (m_use_leadlag) ? 2 : 1;
}
string LeadLaggableChannel::label_suffix(dimn_t variant_no) const
{
    if (m_use_leadlag) {
        if (variant_no == 0) {
            return ":lead";
        } else if (variant_no == 1){
            return ":lag";
        }
        RPY_THROW(std::invalid_argument, "variant is not valid for a lead-lag channel");
    }
    return StreamChannel::label_suffix(variant_no);
}
dimn_t LeadLaggableChannel::variant_id_of_label(string_view label) const
{
    if (m_use_leadlag) {
        if (label == "lead") {
            return 0;
        } else if (label == "lag") {
            return 1;
        }
    }
    return StreamChannel::variant_id_of_label(label);
}
const std::vector<string>& LeadLaggableChannel::get_variants() const
{
    static const std::vector<string> leadlag { "lead", "lag" };
    return (m_use_leadlag) ? leadlag : StreamChannel::get_variants();
}
void LeadLaggableChannel::set_lead_lag(bool new_value)
{
    m_use_leadlag = new_value;
}
bool LeadLaggableChannel::is_lead_lag() const
{
    return m_use_leadlag;
}

#define RPY_EXPORT_MACRO ROUGHPY_STREAMS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::LeadLaggableChannel
#define RPY_SERIAL_DO_REGISTER
#include <roughpy/platform/serialization_instantiations.inl>


RPY_SERIAL_DYNAMIC_INIT(lead_laggable_channel)
