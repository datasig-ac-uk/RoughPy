#ifndef ROUGHPY_STREAMS_STREAM_CONSTRUCTION_HELPER_H_
#define ROUGHPY_STREAMS_STREAM_CONSTRUCTION_HELPER_H_

#include "schema.h"

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <roughpy/algebra/context.h>
#include <roughpy/algebra/lie.h>

#include <boost/container/flat_map.hpp>

namespace rpy {
namespace streams {

class StreamConstructionHelper
{
    std::shared_ptr<StreamSchema> p_schema;
    algebra::context_pointer p_ctx;
    algebra::VectorType m_vtype RPY_UNUSED_VAR = algebra::VectorType::Sparse;

    using multimap_type
            = boost::container::flat_multimap<param_t, algebra::Lie>;

    multimap_type m_entries;
    algebra::Lie m_zero;

    std::vector<key_type> m_dense_keys;
    std::vector<scalars::Scalar> m_previous_values;

public:
    StreamConstructionHelper(algebra::context_pointer ctx,
                             std::shared_ptr<StreamSchema> schema,
                             algebra::VectorType vtype);

private:
    algebra::Lie& current() { return m_entries.rbegin()->second; }
    algebra::Lie& previous()
    {
        if (RPY_UNLIKELY(m_entries.size() < 2)) { return m_zero; }
        return (++m_entries.rbegin())->second;
    }

    algebra::Lie& next_entry(param_t next_timestamp);

public:
    StreamSchema& schema() noexcept { return *p_schema; }
    const scalars::ScalarType* ctype() const noexcept { return p_ctx->ctype(); }

    RPY_NO_DISCARD
    optional<ChannelType> type_of(string_view label) const;

    template <typename T>
    void add_increment(param_t timestamp, dimn_t channel, T&& value);

    template <typename T>
    void add_increment(param_t timestamp, string_view label, T&& value);

    template <typename T>
    void add_value(param_t timestamp, dimn_t channel, T&& value);

    template <typename T>
    void add_value(param_t timestamp, string_view label, T&& value);

    void add_categorical(param_t timestamp, dimn_t channel, dimn_t variant);
    void add_categorical(param_t timestamp, string_view channel,
                         dimn_t variant);
    void add_categorical(param_t timestamp, dimn_t channel,
                         string_view variant);
    void add_categorical(param_t timestmap, string_view channel,
                         string_view variant);

    multimap_type finalise();

    std::shared_ptr<StreamSchema> take_schema() { return p_schema; }
};

template <typename T>
void StreamConstructionHelper::add_increment(param_t timestamp, dimn_t channel,
                                             T&& value)
{
    auto key = static_cast<key_type>(p_schema->channel_to_stream_dim(channel))
            + 1;
    next_entry(timestamp)[key] += scalars::Scalar(std::forward<T>(value));
}
template <typename T>
void StreamConstructionHelper::add_increment(param_t timestamp,
                                             string_view label, T&& value)
{
    auto key = static_cast<key_type>(
                       p_schema->label_to_stream_dim(string(label)))
            + 1;
    next_entry(timestamp)[key] += scalars::Scalar(std::forward<T>(value));
}
template <typename T>
void StreamConstructionHelper::add_value(param_t timestamp, dimn_t channel,
                                         T&& value)
{
    auto lead_idx = p_schema->channel_variant_to_stream_dim(channel, 0);
    auto lag_idx = p_schema->channel_variant_to_stream_dim(channel, 1);

    scalars::Scalar current(std::forward<T>(value));

    next_entry(timestamp)[static_cast<key_type>(lead_idx + 1)]
            += current - m_previous_values[lead_idx];
    next_entry(timestamp)[static_cast<key_type>(lag_idx + 1)]
            += current - m_previous_values[lag_idx];

    m_previous_values[lead_idx] = current;
    m_previous_values[lag_idx] = current;
}
template <typename T>
void StreamConstructionHelper::add_value(param_t timestamp, string_view label,
                                         T&& value)
{
    const auto found = p_schema->find(string(label));
    RPY_CHECK(found != p_schema->end());

    auto lead_idx = found->second.variant_id_of_label("lead");
    auto lag_idx = found->second.variant_id_of_label("lag");

    scalars::Scalar current(std::forward<T>(value));

    next_entry(timestamp)[static_cast<key_type>(lead_idx + 1)]
            += current - m_previous_values[lead_idx];
    next_entry(timestamp)[static_cast<key_type>(lag_idx + 1)]
            += current - m_previous_values[lag_idx];

    m_previous_values[lead_idx] = current;
    m_previous_values[lag_idx] = current;
}

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_STREAM_CONSTRUCTION_HELPER_H_
