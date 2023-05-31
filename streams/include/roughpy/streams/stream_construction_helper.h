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

class StreamConstructionHelper {
    std::shared_ptr<StreamSchema> p_schema;
    algebra::context_pointer p_ctx;
    algebra::VectorType m_vtype = algebra::VectorType::Sparse;
    boost::container::flat_multimap<param_t, algebra::Lie> m_entries;
    algebra::Lie m_zero;

    std::vector<key_type> m_dense_keys;
    std::vector<scalars::Scalar> m_previous_values;

public:

    StreamConstructionHelper(algebra::context_pointer ctx,
                             std::shared_ptr<StreamSchema> schema,
                             dimn_t num_entries,
                             algebra::VectorType vtype
                             );

private:

    algebra::Lie& current() { return m_entries.rbegin()->second; }
    algebra::Lie& previous() {
        if (RPY_UNLIKELY(m_entries.size() < 2)) {
            return m_zero;
        }
        return (++m_entries.rbegin())->second;
    }

public:
    algebra::Lie& next_entry(param_t next_timestamp);


    template <typename T>
    void add_increment(dimn_t channel, T &&value);

    template <typename T>
    void add_increment(string_view label, T &&value);

    template <typename T>
    void add_value(dimn_t channel, T &&value);

    template <typename T>
    void add_value(string_view label, T &&value);

    template <typename T>
    void add_categorical(dimn_t channel, T &&value);

    template <typename T>
    void add_categorical(string_view label, T &&value);

    void add_categorical(dimn_t channel, dimn_t variant);
    void add_categorical(string_view channel, dimn_t variant);
    void add_categorical(dimn_t channel, string_view variant);
    void add_categorical(string_view channel, string_view variant);

    scalars::KeyScalarArray finalise() &&;
};
}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_STREAM_CONSTRUCTION_HELPER_H_
