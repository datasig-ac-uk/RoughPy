// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 24/05/23.
//

#include <roughpy/core/macros.h>
#include <roughpy/streams/schema.h>

#include <algorithm>

using namespace rpy;
using namespace rpy::streams;

StreamChannel::StreamChannel()
    : m_type(ChannelType::Increment), increment_info()
{}
StreamChannel::StreamChannel(const StreamChannel &arg) : m_type(arg.m_type)
{
    switch (m_type) {
        case ChannelType::Increment:
            inplace_construct(&increment_info, arg.increment_info);
            break;
        case ChannelType::Value:
            inplace_construct(&value_info, arg.value_info);
            break;
        case ChannelType::Categorical:
            inplace_construct(&categorical_info, arg.categorical_info);
            break;
        case ChannelType::Lie:
            inplace_construct(&lie_info, arg.lie_info);
            break;
    }
}
StreamChannel::StreamChannel(StreamChannel &&arg) noexcept : m_type(arg.m_type)
{
    switch (m_type) {
        case ChannelType::Increment:
            inplace_construct(&increment_info, arg.increment_info);
            break;
        case ChannelType::Value:
            inplace_construct(&value_info, arg.value_info);
            break;
        case ChannelType::Categorical:
            inplace_construct(&categorical_info, arg.categorical_info);
            break;
        case ChannelType::Lie:
            inplace_construct(&lie_info, arg.lie_info);
            break;
    }
}

StreamChannel::StreamChannel(ChannelType type) : m_type(type)
{
    switch (m_type) {
        case ChannelType::Increment:
            inplace_construct(&increment_info, IncrementChannelInfo());
            break;
        case ChannelType::Value:
            inplace_construct(&value_info, ValueChannelInfo());
            break;
        case ChannelType::Categorical:
            inplace_construct(&categorical_info, CategoricalChannelInfo());
            break;
        case ChannelType::Lie:
            inplace_construct(&lie_info, LieChannelInfo());
            break;
    }
}

StreamChannel &StreamChannel::operator=(const StreamChannel &other)
{
    if (&other != this) {
        this->~StreamChannel();
        m_type = other.m_type;
        switch (m_type) {
            case ChannelType::Increment:
                inplace_construct(&increment_info, other.increment_info);
                break;
            case ChannelType::Value:
                inplace_construct(&value_info, other.value_info);
                break;
            case ChannelType::Categorical:
                inplace_construct(&categorical_info, other.categorical_info);
                break;
            case ChannelType::Lie:
                inplace_construct(&lie_info, other.lie_info);
                break;
        }
    }
    return *this;
}
StreamChannel &StreamChannel::operator=(StreamChannel &&other) noexcept
{
    if (&other != this) {
        this->~StreamChannel();
        m_type = other.m_type;
        switch (m_type) {
            case ChannelType::Increment:
                inplace_construct(&increment_info,
                                  std::move(other.increment_info));
                break;
            case ChannelType::Value:
                inplace_construct(&value_info, std::move(other.value_info));
                break;
            case ChannelType::Categorical:
                inplace_construct(&categorical_info,
                                  std::move(other.categorical_info));
                break;
            case ChannelType::Lie:
                inplace_construct(&lie_info, std::move(other.lie_info));
                break;
        }
    }
    return *this;
}

StreamSchema::StreamSchema(dimn_t width)
{
    reserve(width);
    for (dimn_t i = 0; i < width; ++i) { insert_increment(std::to_string(i)); }
}

StreamChannel::~StreamChannel()
{
    switch (m_type) {
        case ChannelType::Increment:
            increment_info.~IncrementChannelInfo();
            break;
        case ChannelType::Value: value_info.~ValueChannelInfo(); break;
        case ChannelType::Categorical:
            categorical_info.~CategoricalChannelInfo();
            break;
        case ChannelType::Lie: lie_info.~LieChannelInfo(); break;
    }
}

string StreamChannel::label_suffix(dimn_t variant_no) const
{
    switch (m_type) {
        case ChannelType::Increment: return "";
        case ChannelType::Value:
            if (value_info.lead_lag) {
                RPY_CHECK(variant_no < 2);
                return (variant_no == 0) ? ":lead" : ":lag";
            } else {
                return "";
            };
        case ChannelType::Categorical:
            RPY_CHECK(variant_no < categorical_info.variants.size());
            return ":" + categorical_info.variants[variant_no];
        case ChannelType::Lie:
            RPY_CHECK(variant_no < static_cast<dimn_t>(lie_info.width));
            return ":" + std::to_string(variant_no + 1);
    }
    RPY_UNREACHABLE();
}

void StreamChannel::set_lie_info(deg_t width, deg_t depth,
                                 algebra::VectorType vtype)
{
    RPY_CHECK(m_type == ChannelType::Lie);
    lie_info.width = width;
    lie_info.depth = depth;
    lie_info.vtype = vtype;
}

dimn_t StreamChannel::variant_id_of_label(string_view label) const
{
    switch (m_type) {
        case ChannelType::Increment: return 0;
        case ChannelType::Value:
            if (value_info.lead_lag) {
                if (label == "lead") {
                    return 0;
                } else if (label == "lag") {
                    return 1;
                } else {
                    throw std::runtime_error(
                            "unrecognised variant label for type value");
                }
            } else {
                return 0;
            }
        case ChannelType::Categorical: break;
        case ChannelType::Lie:
            deg_t i = std::stoi(string(label));
            RPY_CHECK(i < lie_info.width);
            return i;
    }

    auto it = std::find(categorical_info.variants.begin(),
                        categorical_info.variants.end(), label);
    if (it == categorical_info.variants.end()) {
        throw std::runtime_error(
                "unrecognised variant label for type categorical");
    }

    return static_cast<dimn_t>(it - categorical_info.variants.begin());
}

StreamChannel &StreamChannel::add_variant(string variant_label)
{
    RPY_CHECK(m_type == ChannelType::Categorical);

    if (variant_label.empty()) {
        variant_label = std::to_string(categorical_info.variants.size());
    }

    auto found = std::find(categorical_info.variants.begin(),
                           categorical_info.variants.end(), variant_label);
    if (found != categorical_info.variants.end()) {
        throw std::runtime_error("variant with label " + variant_label
                                 + " already exists");
    }

    categorical_info.variants.push_back(std::move(variant_label));
    return *this;
}
StreamChannel &StreamChannel::insert_variant(string variant_label)
{
    RPY_CHECK(m_type == ChannelType::Categorical);

    if (variant_label.empty()) {
        variant_label = std::to_string(categorical_info.variants.size());
    }

    auto var_begin = categorical_info.variants.begin();
    auto var_end = categorical_info.variants.end();

    auto found = std::find(var_begin, var_end, variant_label);
    if (found == var_end) {
        categorical_info.variants.push_back(std::move(variant_label));
    }

    return *this;
}

std::vector<string> StreamChannel::get_variants() const
{
    std::vector<string> variants;
    switch (m_type) {
        case ChannelType::Increment: break;
        case ChannelType::Value:
            if (value_info.lead_lag) {
                variants.push_back("lead");
                variants.push_back("lag");
            }
            break;
        case ChannelType::Categorical:
            variants = categorical_info.variants;
            break;
        case ChannelType::Lie:
            variants.reserve(lie_info.width);
            for (deg_t i = 0; i < lie_info.width; ++i) {
                variants.push_back(std::to_string(i));
            }
            break;
    }
    return variants;
}

bool StreamSchema::compare_labels(string_view item_label,
                                  string_view ref_label) noexcept
{

    // If the reference label is shorter than item_label it cannot
    // have item_label as a prefix.
    if (ref_label.size() < item_label.size()) { return false; }

    if (ref_label.empty()) { return false; }

    if (item_label.empty()) { return false; }

    auto lit = item_label.begin();
    auto rit = ref_label.begin();

    for (; *lit != '\0'; ++lit, ++rit) {
        if (*rit != *lit) { return false; }
    }

    // Either item_label == ref_label or, ref_label has
    // item_label as a prefix followed by ':'
    return *rit == '\0' || *rit == ':';
}

dimn_t StreamSchema::channel_it_to_width(const_iterator channel_it) const
{
    return channel_it->second.num_variants();
}

dimn_t StreamSchema::width_to_iterator(const_iterator end) const
{
    auto it = begin();

    dimn_t ndims = 0;
    for (; it != end; ++it) { ndims += channel_it_to_width(it); }
    return ndims;
}
typename StreamSchema::const_iterator
StreamSchema::stream_dim_to_channel_it(dimn_t &stream_dim) const
{
    for (auto cit = begin(); cit != end(); ++cit) {
        const auto channel_width = channel_it_to_width(cit);
        if (stream_dim < channel_width) { return cit; }
        stream_dim -= channel_width;
    }
    throw std::runtime_error("stream dimension exceeds width");
}

typename StreamSchema::const_iterator
StreamSchema::find(const string &label) const
{
    auto it_current = begin();
    const auto it_end = end();

    for (; it_current != it_end; ++it_current) {
        if (compare_labels(it_current->first, label)) { return it_current; }
    }

    return it_end;
}

typename StreamSchema::iterator StreamSchema::find(const string &label)
{
    RPY_CHECK(!m_is_final);
    auto it_current = begin();
    const auto it_end = end();

    for (; it_current != it_end; ++it_current) {
        if (compare_labels(it_current->first, label)) { return it_current; }
    }

    return it_end;
}

dimn_t StreamSchema::width() const { return width_to_iterator(end()); }

dimn_t StreamSchema::channel_to_stream_dim(dimn_t channel_no) const
{
    RPY_CHECK(channel_no < size());
    return width_to_iterator(nth(channel_no));
}
dimn_t StreamSchema::channel_variant_to_stream_dim(dimn_t channel_no,
                                                   dimn_t variant_no) const
{
    RPY_CHECK(channel_no < size());
    auto it = nth(channel_no);

    auto so_far = width_to_iterator(it);
    RPY_CHECK(variant_no < it->second.num_variants());
    return so_far + variant_no;
}

std::pair<dimn_t, dimn_t>
StreamSchema::stream_dim_to_channel(dimn_t stream_dim) const
{
    std::pair<dimn_t, dimn_t> result(stream_dim, stream_dim);
    const auto channel_it = stream_dim_to_channel_it(result.second);
    result.first = static_cast<dimn_t>(channel_it - begin());
    return result;
}

string StreamSchema::label_from_channel_it(const_iterator channel_it,
                                           dimn_t variant_id)
{
    return channel_it->first + channel_it->second.label_suffix(variant_id);
}

string StreamSchema::label_of_stream_dim(dimn_t stream_dim) const
{
    auto variant_id = stream_dim;
    auto channel_it = stream_dim_to_channel_it(variant_id);
    return label_from_channel_it(channel_it, variant_id);
}
string_view StreamSchema::label_of_channel_id(dimn_t channel_id) const
{
    RPY_CHECK(channel_id < size());
    return nth(channel_id)->first;
}
string StreamSchema::label_of_channel_variant(dimn_t channel_id,
                                              dimn_t channel_variant) const
{
    RPY_CHECK(channel_id < size());
    return label_from_channel_it(nth(channel_id), channel_variant);
}

dimn_t StreamSchema::label_to_stream_dim(const string &label) const
{
    auto channel = find(label);
    RPY_CHECK(channel != end());

    auto result = width_to_iterator(channel);
    auto variant_begin
            = label.begin() + static_cast<idimn_t>(channel->first.size());
    /*
     * *variant_begin can be either '\0', so the channel is the id
     * we're looking for, or ':', in which case we need to look for
     * a variant.
     */
    switch (*variant_begin) {
        case '\0': return result;
        case ':': ++variant_begin; break;
        default:
            // The find method will not have matched if neither
            // of these cases occurred.
            RPY_UNREACHABLE();
    }

    const string_view variant_label(
            &*variant_begin, static_cast<dimn_t>(label.end() - variant_begin));
    result += channel->second.variant_id_of_label(variant_label);
    return result;
}

StreamChannel &StreamSchema::insert(string label, StreamChannel &&channel_data)
{
    RPY_CHECK(!m_is_final);
    if (label.empty()) { label = std::to_string(size()); }

    // Some users might have labelled their channels using numbers.
    // Silly, but handle it gracefully.

    auto pos = find(label);
    if (pos != end()) { return pos->second; }

    return base_type::insert(pos, {std::move(label), std::move(channel_data)})
            ->second;
}

StreamChannel &StreamSchema::insert(StreamChannel &&channel_data)
{
    return insert(std::to_string(width()), std::move(channel_data));
}

StreamChannel &StreamSchema::insert_increment(string label)
{
    return insert(std::move(label), StreamChannel(IncrementChannelInfo()));
}
StreamChannel &StreamSchema::insert_value(string label)
{
    return insert(std::move(label), StreamChannel(ValueChannelInfo()));
}
StreamChannel &StreamSchema::insert_categorical(string label)
{
    return insert(std::move(label), StreamChannel(CategoricalChannelInfo()));
}
StreamChannel &StreamSchema::insert_lie(string label)
{
    return insert(std::move(label), StreamChannel(LieChannelInfo()));
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::StreamChannel
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_IMPL_CLASSNAME IncrementChannelInfo

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_IMPL_CLASSNAME ValueChannelInfo

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_IMPL_CLASSNAME CategoricalChannelInfo

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_IMPL_CLASSNAME LieChannelInfo

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::StreamSchema

#include <roughpy/platform/serialization_instantiations.inl>
