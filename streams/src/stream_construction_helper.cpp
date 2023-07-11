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
// Created by user on 30/05/23.
//

#include <roughpy/streams/stream_construction_helper.h>

using namespace rpy;
using namespace rpy::streams;

StreamConstructionHelper::StreamConstructionHelper(
        algebra::context_pointer ctx, std::shared_ptr<StreamSchema> schema,
        algebra::VectorType vtype)
    : p_schema(std::move(schema)), p_ctx(std::move(ctx)), m_vtype(vtype),
      m_zero(p_ctx->zero_lie(vtype))
{

    const auto width = p_schema->width();
    m_previous_values.resize(width);
    m_dense_keys.reserve(width);
    for (key_type k = 1; k <= width; ++k) { m_dense_keys.push_back(k); }
}

algebra::Lie& StreamConstructionHelper::next_entry(param_t next_timestamp)
{
    return m_entries.insert({next_timestamp, m_zero})->second;
}

void StreamConstructionHelper::add_categorical(param_t timestamp,
                                               dimn_t channel, dimn_t variant)
{
    auto idx = p_schema->channel_variant_to_stream_dim(channel, variant);
    auto key = static_cast<key_type>(idx + 1);
    next_entry(timestamp)[key] += p_ctx->ctype()->one();
}
void StreamConstructionHelper::add_categorical(param_t timestamp,
                                               string_view channel,
                                               dimn_t variant)
{
    const auto found = p_schema->find(string(channel));
    RPY_CHECK(found != p_schema->end());
    RPY_CHECK(variant < found->second.num_variants());
    auto key = static_cast<key_type>(found - p_schema->begin())
            + static_cast<key_type>(variant) + 1;
    next_entry(timestamp)[key] += p_ctx->ctype()->one();
}
void StreamConstructionHelper::add_categorical(param_t timestamp,
                                               dimn_t channel,
                                               string_view variant)
{
    RPY_CHECK(channel < p_schema->size());
    const auto channel_item = p_schema->nth(channel);

    const auto variants = channel_item->second.get_variants();
    const auto found = std::find(variants.begin(), variants.end(), variant);
    RPY_CHECK(found != variants.end());

    auto key = static_cast<key_type>(p_schema->channel_variant_to_stream_dim(
            channel, static_cast<dimn_t>(found - variants.begin())));
    next_entry(timestamp)[key] += p_ctx->ctype()->one();
}
void StreamConstructionHelper::add_categorical(param_t timestamp,
                                               string_view channel,
                                               string_view variant)
{
    auto idx = p_schema->label_to_stream_dim(string(channel) + ':'
                                             + string(variant));
    next_entry(timestamp)[static_cast<key_type>(idx + 1)]
            += p_ctx->ctype()->one();
}
typename StreamConstructionHelper::multimap_type
StreamConstructionHelper::finalise()
{
    boost::container::flat_multimap<param_t, algebra::Lie> result;
    result = std::move(m_entries);
    return result;
}

optional<ChannelType> StreamConstructionHelper::type_of(string_view label) const
{
    const auto& schema = *p_schema;
    auto found = schema.find(string(label));
    if (found != schema.end()) { return found->second.type(); }
    return {};
}
