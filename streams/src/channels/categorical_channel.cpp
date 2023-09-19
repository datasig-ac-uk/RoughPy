//
// Created by sam on 07/08/23.
//

#include <roughpy/streams/channels/categorical_channel.h>

#include <algorithm>

using namespace rpy;
using namespace rpy::streams;


dimn_t CategoricalChannel::num_variants() const
{
    return m_variants.size();
}
string CategoricalChannel::label_suffix(dimn_t variant_no) const
{
    RPY_CHECK(variant_no < m_variants.size());
    return ":" + m_variants[variant_no];
}
dimn_t CategoricalChannel::variant_id_of_label(string_view label) const
{
    auto it = std::find(m_variants.begin(), m_variants.end(), label);
    if (it == m_variants.end()) {
        RPY_THROW(std::runtime_error,
                "unrecognised variant label for type categorical");
    }

    return static_cast<dimn_t>(it - m_variants.begin());
}

const std::vector<string>& CategoricalChannel::get_variants() const
{
    return m_variants;
}
StreamChannel& CategoricalChannel::add_variant(string variant_label)
{
    string label;
    if (variant_label.empty()) {
        label = std::to_string(m_variants.size());
    } else {
        label = variant_label;
    }

    auto var_begin = m_variants.begin();
    auto var_end = m_variants.end();
    auto found = std::find(var_begin, var_end, label);
    if (found != var_end) {
        RPY_THROW(std::runtime_error,
                  "variant with label " + label + " already exists");
    }
    m_variants.push_back(std::move(label));

    return *this;

}
StreamChannel& CategoricalChannel::insert_variant(string variant_label)
{
    string label;
    if (variant_label.empty()) {
        label = std::to_string(m_variants.size());
    } else {
        label = variant_label;
    }

    auto var_begin = m_variants.begin();
    auto var_end = m_variants.end();
    auto found = std::find(var_begin, var_end, label);
    if (found == var_end) {
        m_variants.push_back(std::move(label));
    }

    return *this;
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::CategoricalChannel
#include <roughpy/platform/serialization_instantiations.inl>