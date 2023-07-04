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
#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/real_interval.h>
#include <roughpy/platform/serialization.h>

#include <boost/container/flat_map.hpp>

#include <functional>
#include <variant>
#include <vector>

#include "roughpy/scalars/key_scalar_array.h"

namespace rpy {
namespace streams {

enum struct ChannelType : uint8_t
{
    Increment = 0,
    Value = 1,
    Categorical = 2,
    Lie = 3
};

struct IncrementChannelInfo {
};
struct ValueChannelInfo {
    bool lead_lag = false;
};

struct CategoricalChannelInfo {
    std::vector<string> variants;
};
struct LieChannelInfo {
    deg_t width;
    deg_t depth;
    algebra::VectorType vtype;
};

/**
 * @brief Abstract description of a single channel of data.
 *
 * A stream channel contains metadata about one particular channel of
 * incoming raw data. The data can take one of four possible forms:
 * increment data, value data, categorical data, or lie data. Each form
 * of data has its own set of required and optional metadata, including
 * the number and labels of a categorical variable. These objects are
 * typically accessed via a schema, which maintains the collection of
 * all channels associated with a stream.
 */
class StreamChannel
{
    ChannelType m_type;
    union
    {
        IncrementChannelInfo increment_info;
        ValueChannelInfo value_info;
        CategoricalChannelInfo categorical_info;
        LieChannelInfo lie_info;
    };

    template <typename T>
    static void inplace_construct(void* address, T&& value) noexcept(
            is_nothrow_constructible<remove_cv_ref_t<T>,
                                     decltype(value)>::value)
    {
        ::new (address) remove_cv_ref_t<T>(std::forward<T>(value));
    }

public:
    RPY_NO_DISCARD
    ChannelType type() const noexcept { return m_type; }

    RPY_NO_DISCARD dimn_t num_variants() const
    {
        switch (m_type) {
            case ChannelType::Increment: return 1;
            case ChannelType::Value: return value_info.lead_lag ? 2 : 1;
            case ChannelType::Categorical:
                return categorical_info.variants.size();
            case ChannelType::Lie: return lie_info.width;
        }
        RPY_UNREACHABLE();
    }

    RPY_NO_DISCARD
    string label_suffix(dimn_t variant_no) const;

    StreamChannel();
    StreamChannel(const StreamChannel& arg);
    StreamChannel(StreamChannel&& arg) noexcept;

    explicit StreamChannel(ChannelType type);

    explicit StreamChannel(IncrementChannelInfo&& info)
        : m_type(ChannelType::Increment), increment_info(std::move(info))
    {}

    explicit StreamChannel(ValueChannelInfo&& info)
        : m_type(ChannelType::Value), value_info(std::move(info))
    {}

    explicit StreamChannel(CategoricalChannelInfo&& info)
        : m_type(ChannelType::Categorical), categorical_info(std::move(info))
    {}

    explicit StreamChannel(LieChannelInfo&& info)
        : m_type(ChannelType::Lie), lie_info(std::move(info))
    {}

    ~StreamChannel();

    StreamChannel& operator=(const StreamChannel& other);
    StreamChannel& operator=(StreamChannel&& other) noexcept;

    StreamChannel& operator=(IncrementChannelInfo&& info)
    {
        this->~StreamChannel();
        m_type = ChannelType::Increment;
        inplace_construct(&increment_info, std::move(info));
        return *this;
    }

    StreamChannel& operator=(ValueChannelInfo&& info)
    {
        this->~StreamChannel();
        m_type = ChannelType::Value;
        inplace_construct(&value_info, std::move(info));
        return *this;
    }

    StreamChannel& operator=(CategoricalChannelInfo&& info)
    {
        this->~StreamChannel();
        m_type = ChannelType::Categorical;
        inplace_construct(&categorical_info, std::move(info));
        return *this;
    }

    RPY_NO_DISCARD
    dimn_t variant_id_of_label(string_view label) const;

    void set_lie_info(deg_t width, deg_t depth, algebra::VectorType vtype);
    void set_lead_lag(bool new_value)
    {
        RPY_CHECK(m_type == ChannelType::Value);
        value_info.lead_lag = new_value;
    }
    RPY_NO_DISCARD
    bool is_lead_lag() const
    {
        RPY_CHECK(m_type == ChannelType::Value);
        return value_info.lead_lag;
    }

    StreamChannel& add_variant(string variant_label);

    /**
     * @brief Insert a new variant if it doesn't already exist
     * @param variant_label variant label to insert.
     * @return referene to this channel.
     */
    StreamChannel& insert_variant(string variant_label);

    RPY_NO_DISCARD
    std::vector<string> get_variants() const;

    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();
};

class RPY_EXPORT SchemaContext
{
    param_t m_param_offset = 0.0;
    param_t m_param_scaling = 1.0;

public:
    virtual ~SchemaContext();

    RPY_NO_DISCARD
    intervals::RealInterval
    reparametrize(const intervals::RealInterval& arg) const
    {
        return {m_param_offset + m_param_scaling * arg.inf(),
                m_param_offset + m_param_scaling * arg.sup(), arg.type()};
    }

    RPY_NO_DISCARD
    virtual intervals::RealInterval
    convert_parameter_interval(const intervals::Interval& arg) const;
};

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

    bool m_is_final = false;

    std::unique_ptr<SchemaContext> p_context;

public:
    using typename base_type::const_iterator;
    using typename base_type::iterator;
    using typename base_type::value_type;

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

    RPY_NO_DISCARD
    bool is_final() const noexcept { return m_is_final; }
    void finalize() noexcept { m_is_final = true; }

    RPY_NO_DISCARD
    intervals::RealInterval
    adjust_interval(const intervals::Interval& arg) const;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_EXT(IncrementChannelInfo)
{
    (void) value;
    RPY_SERIAL_SERIALIZE_NVP("data", 0);
}

RPY_SERIAL_SERIALIZE_FN_EXT(ValueChannelInfo)
{
    RPY_SERIAL_SERIALIZE_NVP("lead_lag", value.lead_lag);
}

RPY_SERIAL_SERIALIZE_FN_EXT(CategoricalChannelInfo)
{
    RPY_SERIAL_SERIALIZE_NVP("variants", value.variants);
}

RPY_SERIAL_SERIALIZE_FN_EXT(LieChannelInfo)
{
    RPY_SERIAL_SERIALIZE_NVP("width", value.width);
    RPY_SERIAL_SERIALIZE_NVP("depth", value.depth);
    RPY_SERIAL_SERIALIZE_NVP("vector_type", value.vtype);
}

RPY_SERIAL_SAVE_FN_IMPL(StreamChannel)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    switch (m_type) {
        case ChannelType::Increment:
            RPY_SERIAL_SERIALIZE_NVP("increment", increment_info);
            break;
        case ChannelType::Value:
            RPY_SERIAL_SERIALIZE_NVP("value", value_info);
            break;
        case ChannelType::Categorical:
            RPY_SERIAL_SERIALIZE_NVP("categorical", categorical_info);
            break;
        case ChannelType::Lie: RPY_SERIAL_SERIALIZE_NVP("lie", lie_info); break;
    }
}

RPY_SERIAL_LOAD_FN_IMPL(StreamChannel)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    switch (m_type) {
        case ChannelType::Increment:
            new (&increment_info) IncrementChannelInfo;
            RPY_SERIAL_SERIALIZE_NVP("increment", increment_info);
            break;
        case ChannelType::Value:
            new (&value_info) ValueChannelInfo;
            RPY_SERIAL_SERIALIZE_NVP("value", value_info);
            break;
        case ChannelType::Categorical:
            new (&categorical_info) CategoricalChannelInfo;
            RPY_SERIAL_SERIALIZE_NVP("categorical", categorical_info);
            break;
        case ChannelType::Lie:
            new (&lie_info) LieChannelInfo;
            RPY_SERIAL_SERIALIZE_NVP("lie", lie_info);
            break;
    }
}

RPY_SERIAL_SERIALIZE_FN_IMPL(StreamSchema)
{
    RPY_SERIAL_SERIALIZE_NVP("channels", static_cast<base_type&>(*this));
    RPY_SERIAL_SERIALIZE_NVP("is_final", m_is_final);
}

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_SCHEMA_H
