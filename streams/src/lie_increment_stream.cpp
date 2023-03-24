//
// Created by user on 10/03/23.
//

#include "lie_increment_stream.h"


using namespace rpy;
using namespace rpy::streams;

LieIncrementStream::LieIncrementStream(
    scalars::KeyScalarArray &&buffer,
    Slice<param_t> indices,
    StreamMetadata metadata)
    : base_t(std::move(metadata)),
      m_buffer(std::move(buffer))
{
    const auto &md = this->metadata();
    for (dimn_t i = 0; i < indices.size(); ++i) {
        m_mapping[indices[i]] = i * md.width;
    }
}

algebra::Lie LieIncrementStream::log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const {

    const auto &md = metadata();
    if (empty(interval)) {
        return ctx.zero_lie(md.cached_vector_type);
    }

    rpy::algebra::SignatureData data{
        scalars::ScalarStream(ctx.ctype()),
        {},
        md.cached_vector_type};

    if (m_mapping.size() == 1) {
        data.data_stream.set_elts_per_row(m_buffer.size());
    } else if (m_mapping.size() > 1) {
        auto row1 = (++m_mapping.begin())->second;
        auto row0 = m_mapping.begin()->second;
        data.data_stream.set_elts_per_row(row1 - row0);
    }

    auto begin = (interval.type() == intervals::IntervalType::Opencl)
                 ? m_mapping.lower_bound(interval.inf())
                 : m_mapping.upper_bound(interval.inf());

    auto end = (interval.type() == intervals::IntervalType::Opencl)
               ? m_mapping.lower_bound(interval.sup())
               : m_mapping.upper_bound(interval.sup());

    if (begin == end) {
        return ctx.zero_lie(md.cached_vector_type);
    }

    data.data_stream.reserve_size(end - begin);

    for (auto it1 = begin, it = it1++; it1 != end; ++it, ++it1) {
        data.data_stream.push_back({m_buffer[it->second].to_pointer(), it1->second - it->second});
    }
    // Case it = it1 - 1 and it1 == end
    --end;
    data.data_stream.push_back({m_buffer[end->second].to_pointer(), m_buffer.size() - end->second});

    if (m_buffer.keys()!= nullptr) {
        data.key_stream.reserve(end - begin);
        ++end;
        for (auto it = begin; it != end; ++it) {
            data.key_stream.push_back(m_buffer.keys() + it->second);
        }
    }

    assert(ctx.width() == md.width);
    //    assert(ctx.depth() == md.depth);

    return ctx.log_signature(data);
}
bool LieIncrementStream::empty(const intervals::Interval &interval) const noexcept {

    auto begin = (interval.type() == intervals::IntervalType::Opencl)
                 ? m_mapping.lower_bound(interval.inf())
                 : m_mapping.upper_bound(interval.inf());

    auto end = (interval.type() == intervals::IntervalType::Opencl)
               ? m_mapping.lower_bound(interval.sup())
               : m_mapping.upper_bound(interval.sup());

    return begin == end;
}
