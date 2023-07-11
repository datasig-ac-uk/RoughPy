// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_
#define ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_

#include "stream_base.h"

#include <map>
#include <mutex>

#include <roughpy/platform/serialization.h>

namespace rpy {
namespace streams {

class DataIncrement
{
    using Lie = algebra::Lie;
    using DyadicInterval = intervals::DyadicInterval;

public:
    using map_type = std::map<DyadicInterval, DataIncrement>;
    using data_increment = typename map_type::iterator;
    using const_data_increment = typename map_type::const_iterator;

private:
    resolution_t m_accuracy;
    Lie m_lie;
    data_increment m_sibling = data_increment();
    data_increment m_parent = data_increment();

public:
    DataIncrement() : m_accuracy(-1), m_lie(), m_sibling(), m_parent() {}

    DataIncrement(Lie&& val, resolution_t accuracy,
                  data_increment sibling = data_increment(),
                  data_increment parent = data_increment())
        : m_accuracy(accuracy), m_lie(std::move(val)), m_sibling(sibling),
          m_parent(parent)
    {}

    void lie(Lie&& new_lie) noexcept { m_lie = std::move(new_lie); }
    RPY_NO_DISCARD
    const Lie& lie() const noexcept { return m_lie; }

    void accuracy(resolution_t new_accuracy) noexcept
    {
        m_accuracy = new_accuracy;
    }
    RPY_NO_DISCARD
    resolution_t accuracy() const noexcept { return m_accuracy; }

    void sibling(data_increment sib) { m_sibling = sib; }
    RPY_NO_DISCARD
    data_increment sibling() const { return m_sibling; }
    void parent(data_increment par) { m_parent = par; }
    RPY_NO_DISCARD
    data_increment parent() const { return m_parent; }

    RPY_NO_DISCARD
    static bool is_leaf(data_increment increment) noexcept
    {
        return increment->first.power() == increment->second.accuracy();
    }
};

class DynamicallyConstructedStream : public StreamInterface
{
public:
    using DyadicInterval = intervals::DyadicInterval;
    using Lie = algebra::Lie;

protected:
    using data_tree_type = typename DataIncrement::map_type;
    using data_increment = typename data_tree_type::iterator;
    using const_data_increment = typename data_tree_type::const_iterator;

private:
    mutable std::recursive_mutex m_lock;
    mutable data_tree_type m_data_tree;

    void refine_accuracy(data_increment increment, resolution_t desired) const;

    RPY_NO_DISCARD
    data_increment expand_root_until_contains(data_increment root,
                                              DyadicInterval di) const;

    RPY_NO_DISCARD
    data_increment insert_node(DyadicInterval di, Lie&& value,
                               resolution_t accuracy,
                               data_increment hint) const;

    RPY_NO_DISCARD
    data_increment insert_children_and_refine(data_increment leaf,
                                              DyadicInterval interval) const;

    data_increment update_parent_accuracy(data_increment below) const;

    void update_parents(data_increment current) const;

protected:
    /// Safely update the given increment with the new values and accuracy
    void update_increment(data_increment increment, Lie&& new_value,
                          resolution_t resolution) const;

    /// Safely get the lie value associated with an increment
    const Lie& lie_value(const_data_increment increment) noexcept;

    RPY_NO_DISCARD
    virtual Lie make_new_root_increment(DyadicInterval di) const;
    RPY_NO_DISCARD
    virtual Lie
    make_neighbour_root_increment(DyadicInterval neighbour_di) const;
    RPY_NO_DISCARD
    virtual pair<Lie, Lie>
    compute_child_lie_increments(DyadicInterval left_di,
                                 DyadicInterval right_di,
                                 const Lie& parent_value) const;

public:
    using StreamInterface::StreamInterface;

    DynamicallyConstructedStream(DynamicallyConstructedStream&& other) noexcept
        : StreamInterface(static_cast<StreamInterface&&>(other)), m_lock(),
          m_data_tree(std::move(other.m_data_tree))
    {}

    algebra::Lie log_signature(const intervals::DyadicInterval& interval,
                               resolution_t resolution,
                               const algebra::Context& ctx) const override;
    algebra::Lie log_signature(const intervals::Interval& domain,
                               resolution_t resolution,
                               const algebra::Context& ctx) const override;

protected:
    template <typename Archive>
    void store_cache(Archive& archive) const;

    template <typename Archive>
    void load_cache(Archive& archive, const algebra::Context& ctx);
};

namespace dtl {

struct DataIncrementSafe {
    intervals::DyadicInterval interval;
    resolution_t resolution;
    algebra::Lie content;
    dimn_t sibling_idx = 0;
    dimn_t parent_idx = 0;
};

RPY_SERIAL_SERIALIZE_FN_EXT(DataIncrementSafe)
{
    RPY_SERIAL_SERIALIZE_NVP("interval", value.interval);
    RPY_SERIAL_SERIALIZE_NVP("resolution", value.resolution);
    RPY_SERIAL_SERIALIZE_NVP("content", value.content);
    RPY_SERIAL_SERIALIZE_NVP("sibling_idx", value.sibling_idx);
    RPY_SERIAL_SERIALIZE_NVP("parent_idx", value.parent_idx);
}

}// namespace dtl

template <typename Archive>
void DynamicallyConstructedStream::store_cache(Archive& archive) const
{
    std::lock_guard<std::recursive_mutex> access(m_lock);
    std::map<DyadicInterval, dimn_t> indices;
    std::vector<dtl::DataIncrementSafe> linear_data;
    linear_data.reserve(m_data_tree.size());

    dimn_t index = 0;
    for (const auto& item : m_data_tree) {
        linear_data.push_back(
                {item.first, item.second.accuracy(), item.second.lie()});
        indices[item.first] = index++;
    }

    for (const auto& item : m_data_tree) {
        auto& entry = linear_data[indices[item.first]];
        entry.parent_idx = indices[item.second.parent()->first];
        entry.sibling_idx = indices[item.second.sibling()->first];
    }

    RPY_SERIAL_SERIALIZE_NVP("cache_data", linear_data);
}
template <typename Archive>
void DynamicallyConstructedStream::load_cache(Archive& archive,
                                              const algebra::Context& ctx)
{
    std::vector<dtl::DataIncrementSafe> linear_data;
    RPY_SERIAL_SERIALIZE_NVP("cache_data", linear_data);

    for (auto& item : linear_data) {
        auto& entry = m_data_tree[item.interval];
        entry.accuracy(item.resolution);
        entry.lie(std::move(item.content));
    }

    for (auto& item : linear_data) {
        auto& entry = m_data_tree[item.interval];
        entry.parent(m_data_tree.find(linear_data[item.parent_idx].interval));
        entry.sibling(m_data_tree.find(linear_data[item.sibling_idx].interval));
    }
}

}// namespace streams
}// namespace rpy

RPY_SERIAL_CLASS_VERSION(rpy::streams::dtl::DataIncrementSafe, 0);

#endif// ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_
