//
// Created by sam on 07/08/23.
//


#include <roughpy/streams/channels/stream_channel.h>


using namespace rpy;
using namespace rpy::streams;

StreamChannel::~StreamChannel() {}

dimn_t StreamChannel::num_variants() const
{
    return 1;
}

string StreamChannel::label_suffix(rpy::dimn_t variant_no) const
{
    return "";
}

dimn_t StreamChannel::variant_id_of_label(string_view label) const { return 0; }
void StreamChannel::set_lie_info(
        deg_t width, deg_t depth, algebra::VectorType vtype
)
{}
StreamChannel& StreamChannel::add_variant(string variant_label)
{
    return *this;
}
StreamChannel& StreamChannel::insert_variant(string variant_label)
{
    return *this;
}
const std::vector<string>& StreamChannel::get_variants() const
{
    static const std::vector<string> no_variants;
    return no_variants;
}

void StreamChannel::set_lead_lag(bool new_value) {}
bool StreamChannel::is_lead_lag() const { return false; }
void StreamChannel::convert_input(
        scalars::ScalarPointer& dst, const scalars::ScalarPointer& src,
        dimn_t count
) const
{
    if (count == 0) { return; }
    RPY_CHECK(!src.is_null());
    RPY_CHECK(dst.type() != nullptr);

    if (dst.is_null()) {
        dst = dst.type()->allocate(count);
    }

    dst.type()->convert_copy(dst, src, count);
}




#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::StreamChannel
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>