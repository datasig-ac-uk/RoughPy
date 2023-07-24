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

#ifndef ROUGHPY_STREAMS_SCHEMA_H
#define ROUGHPY_STREAMS_SCHEMA_H

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/algebra/lie_basis.h>
#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/real_interval.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/key_scalar_array.h>

#include <boost/container/flat_map.hpp>

#include <functional>
#include <variant>
#include <vector>

#include "channels.h"
#include "schema_context.h"

namespace rpy {
namespace streams {

/**
 * @brief An abstract description of the stream data.
 *
 * The schema has two core roles in a stream. The first is to function
 * as a mapping between the incoming data channels and the dimensions
 * of the stream. This is most important when constructing the stream
 * from raw data, but can also come into effect at other points in the
 * process. The second is to store additional metadata and conversion
 * information between raw data values and normalised increment data
 * used by the stream. This includes the mapping of raw time-stamp data
 * from the raw (date)time stamp onto the streams rebased parametrisation.
 *
 * A secondary function of the schema is to calculate the width required
 * of a stream context in order to fully accommodate the input raw data.
 * For simple streams, where each individual channel contains increment
 * data, this is a 1-1 mapping. However, categorical values must be expanded
 * and other types of data might occupy multiple stream dimensions.
 */
class RPY_EXPORT StreamSchema : private std::vector<pair<string, StreamChannel>>
{
    using base_type = std::vector<pair<string, StreamChannel>>;
    using static_vec_type = std::vector<pair<string, StaticChannel>>;

    static_vec_type m_static_channels;

    bool m_is_final = false;

    std::unique_ptr<SchemaContext> p_context;

public:
    using typename base_type::const_iterator;
    using typename base_type::iterator;
    using static_iterator = typename static_vec_type::iterator;
    using static_const_iterator = typename static_vec_type::const_iterator;
    using typename base_type::value_type;
    using lie_key = typename algebra::LieBasis::key_type;

    using base_type::begin;
    using base_type::emplace_back;
    using base_type::end;
    using base_type::reserve;
    using base_type::size;

    static bool compare_labels(string_view item_label,
                               string_view ref_label) noexcept;

private:
    RPY_NO_DISCARD
    dimn_t channel_it_to_width(const_iterator channel_it) const;

    RPY_NO_DISCARD
    dimn_t width_to_iterator(const_iterator end) const;

    /**
     * @brief Get the iterator to the channel corresponding to stream_dim
     * @param stream_dim Stream dimension to look up, modified to be the
     * variant_id after call
     * @return const_iterator to the channel
     */
    const_iterator stream_dim_to_channel_it(dimn_t& stream_dim) const;

public:
    StreamSchema() = default;
    StreamSchema(const StreamSchema&) = delete;
    StreamSchema(StreamSchema&&) noexcept = default;

    explicit StreamSchema(dimn_t width);

    StreamSchema& operator=(const StreamSchema&) = delete;
    StreamSchema& operator=(StreamSchema&&) noexcept = default;

    RPY_NO_DISCARD
    const_iterator nth(dimn_t idx) const noexcept
    {
        RPY_DBG_ASSERT(idx < size());
        return begin() + static_cast<idimn_t>(idx);
    }

    RPY_NO_DISCARD
    SchemaContext* context() const noexcept { return p_context.get(); }

    template <typename Context, typename... Args>
    enable_if_t<is_base_of<SchemaContext, Context>::value>
    init_context(Args&&... args)
    {
        RPY_DBG_ASSERT(!p_context);
        p_context = std::make_unique<Context>(std::forward<Args>(args)...);
    }

public:
    RPY_NO_DISCARD
    iterator find(const string& label);

    RPY_NO_DISCARD
    const_iterator find(const string& label) const;

    RPY_NO_DISCARD
    static_iterator begin_static() noexcept
    {
        return m_static_channels.begin();
    }

    RPY_NO_DISCARD
    static_iterator end_static() noexcept { return m_static_channels.end(); }

    RPY_NO_DISCARD
    static_const_iterator begin_static() const noexcept
    {
        return m_static_channels.cbegin();
    }

    RPY_NO_DISCARD
    static_const_iterator end_static() const noexcept
    {
        return m_static_channels.cend();
    }

    RPY_NO_DISCARD
    static_iterator find_static(const string& label);

    RPY_NO_DISCARD
    static_const_iterator find_static(const string& label) const;

    RPY_NO_DISCARD
    bool contains(const string& label) const { return find(label) != end(); }

    RPY_NO_DISCARD
    dimn_t width() const;

    RPY_NO_DISCARD
    dimn_t channel_to_stream_dim(dimn_t channel_no) const;

    RPY_NO_DISCARD
    dimn_t channel_variant_to_stream_dim(dimn_t channel_no,
                                         dimn_t variant_no) const;

    RPY_NO_DISCARD
    pair<dimn_t, dimn_t> stream_dim_to_channel(dimn_t stream_dim) const;

private:
    RPY_NO_DISCARD
    static string label_from_channel_it(const_iterator channel_it,
                                        dimn_t variant_id);

public:
    RPY_NO_DISCARD
    string label_of_stream_dim(dimn_t stream_dim) const;

    RPY_NO_DISCARD
    string_view label_of_channel_id(dimn_t channel_id) const;

    RPY_NO_DISCARD
    string label_of_channel_variant(dimn_t channel_id,
                                    dimn_t channel_variant) const;

    RPY_NO_DISCARD
    dimn_t label_to_stream_dim(const string& label) const;

    StreamChannel& insert(string label, StreamChannel&& channel_data);

    StreamChannel& insert(StreamChannel&& channel_data);

    StreamChannel& insert_increment(string label);
    StreamChannel& insert_value(string label);
    StreamChannel& insert_categorical(string label);
    StreamChannel& insert_lie(string label);

    StaticChannel& insert_static_value(string label);
    StaticChannel& insert_static_categorical(string label);

    RPY_NO_DISCARD
    bool is_final() const noexcept { return m_is_final; }
    void finalize() noexcept { m_is_final = true; }

    RPY_NO_DISCARD
    intervals::RealInterval
    adjust_interval(const intervals::Interval& arg) const;

    RPY_NO_DISCARD
    lie_key label_to_lie_key(const string& label);

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(StreamSchema)
{
    RPY_SERIAL_SERIALIZE_NVP("channels", static_cast<base_type&>(*this));
    RPY_SERIAL_SERIALIZE_NVP("is_final", m_is_final);
}

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_SCHEMA_H
