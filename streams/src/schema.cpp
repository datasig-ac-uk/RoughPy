// Copyright (c) 2023 Datasig Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 24/05/23.
//

#include <roughpy/core/macros.h>
#include <roughpy/streams/schema.h>

#include <algorithm>

using namespace rpy;
using namespace rpy::streams;

StreamSchema::StreamSchema(dimn_t width)
{
    reserve(width);
    for (dimn_t i = 0; i < width; ++i) { insert_increment(std::to_string(i)); }
}

bool StreamSchema::compare_labels(
        string_view item_label, string_view ref_label
) noexcept
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
StreamSchema::stream_dim_to_channel_it(dimn_t& stream_dim) const
{
    for (auto cit = begin(); cit != end(); ++cit) {
        const auto channel_width = channel_it_to_width(cit);
        if (stream_dim < channel_width) { return cit; }
        stream_dim -= channel_width;
    }
    RPY_THROW(std::runtime_error, "stream dimension exceeds width");
}

typename StreamSchema::const_iterator StreamSchema::find(const string& label
) const
{
    auto it_current = begin();
    const auto it_end = end();

    for (; it_current != it_end; ++it_current) {
        if (compare_labels(it_current->first, label)) { return it_current; }
    }

    return it_end;
}

typename StreamSchema::iterator StreamSchema::find(const string& label)
{
    RPY_CHECK(!m_is_final);
    auto it_current = begin();
    const auto it_end = end();

    for (; it_current != it_end; ++it_current) {
        if (compare_labels(it_current->first, label)) { return it_current; }
    }

    return it_end;
}

typename StreamSchema::static_iterator
StreamSchema::find_static(const string& label)
{
    RPY_CHECK(!m_is_final);
    auto it_current = m_static_channels.begin();
    const auto it_end = m_static_channels.end();

    for (; it_current != it_end; ++it_current) {
        if (compare_labels(it_current->first, label)) { return it_current; }
    }
    return it_end;
}
typename StreamSchema::static_const_iterator
StreamSchema::find_static(const string& label) const
{
    auto it_current = m_static_channels.cbegin();
    const auto it_end = m_static_channels.cend();

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
dimn_t StreamSchema::channel_variant_to_stream_dim(
        dimn_t channel_no, dimn_t variant_no
) const
{
    RPY_CHECK(channel_no < size());
    auto it = nth(channel_no);

    auto so_far = width_to_iterator(it);
    RPY_CHECK(variant_no < it->second.num_variants());
    return so_far + variant_no;
}

std::pair<dimn_t, dimn_t> StreamSchema::stream_dim_to_channel(dimn_t stream_dim
) const
{
    std::pair<dimn_t, dimn_t> result(stream_dim, stream_dim);
    const auto channel_it = stream_dim_to_channel_it(result.second);
    result.first = static_cast<dimn_t>(channel_it - begin());
    return result;
}

string StreamSchema::label_from_channel_it(
        const_iterator channel_it, dimn_t variant_id
)
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
string StreamSchema::label_of_channel_variant(
        dimn_t channel_id, dimn_t channel_variant
) const
{
    RPY_CHECK(channel_id < size());
    return label_from_channel_it(nth(channel_id), channel_variant);
}

dimn_t StreamSchema::label_to_stream_dim(const string& label) const
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
            &*variant_begin, static_cast<dimn_t>(label.end() - variant_begin)
    );
    result += channel->second.variant_id_of_label(variant_label);
    return result;
}

StreamChannel& StreamSchema::insert(string label, StreamChannel&& channel_data)
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

intervals::RealInterval
StreamSchema::adjust_interval(const intervals::Interval& arg) const
{
    if (p_context) { return p_context->convert_parameter_interval(arg); }
    return intervals::RealInterval(arg);
}

StreamChannel& StreamSchema::insert(StreamChannel&& channel_data)
{
    return insert(std::to_string(width()), std::move(channel_data));
}

StreamChannel& StreamSchema::insert_increment(string label)
{
    return insert(std::move(label), StreamChannel(IncrementChannelInfo()));
}
StreamChannel& StreamSchema::insert_value(string label)
{
    return insert(std::move(label), StreamChannel(ValueChannelInfo()));
}
StreamChannel& StreamSchema::insert_categorical(string label)
{
    return insert(std::move(label), StreamChannel(CategoricalChannelInfo()));
}
StreamChannel& StreamSchema::insert_lie(string label)
{
    return insert(std::move(label), StreamChannel(LieChannelInfo()));
}

StaticChannel& StreamSchema::insert_static_value(string label)
{
    auto found = find_static(label);
    auto end = end_static();

    if (found != end) { return found->second; }

    return m_static_channels
            .insert(end, {std::move(label), StaticChannel(ValueChannelInfo())})
            ->second;
}
StaticChannel& StreamSchema::insert_static_categorical(string label)
{

    auto found = find_static(label);
    auto end = end_static();

    if (found != end) { return found->second; }

    return m_static_channels
            .insert(end,
                    {std::move(label), StaticChannel(CategoricalChannelInfo())})
            ->second;
}
typename StreamSchema::lie_key
StreamSchema::label_to_lie_key(const string& label)
{
    auto idx = label_to_stream_dim(label);
    return static_cast<lie_key>(idx) + 1;
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::StreamSchema

#include <roughpy/platform/serialization_instantiations.inl>
