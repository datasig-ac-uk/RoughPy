//
// Created by sammorley on 03/12/24.
//

#include "value_stream.h"


using namespace rpy;
using namespace rpy::streams;


std::shared_ptr<const ValueStream> ValueStream::query(
    const intervals::Interval& interval) const
{
    return std::make_shared<const ValueStream>(
        intervals::RealInterval(interval),
        p_arrival_stream,
        p_increment_stream);
}

bool ValueStream::empty(const intervals::Interval& interval) const noexcept
{
    if (!interval.intersects_with(m_domain)) { return false; }

    return p_arrival_stream->empty(interval) && p_increment_stream->empty(
        interval);
}

algebra::Lie ValueStream::log_signature_impl(
    const intervals::Interval& interval,
    const algebra::Context& ctx) const
{
    auto query_interval = intersection(interval, m_domain);

    return p_increment_stream->log_signature(interval, ctx);
}