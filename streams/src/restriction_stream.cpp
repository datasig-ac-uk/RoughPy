//
// Created by sammorley on 06/12/24.
//

#include "roughpy/streams/restriction_stream.h"

#include "roughpy/intervals/interval.h"


using namespace rpy;
using namespace rpy::streams;



bool rpy::streams::RestrictionStream::empty(
    const intervals::Interval& interval) const noexcept
{
    if (!m_domain.intersects_with(interval)) { return true; }
    const auto query = intersection(m_domain, interval);
    return p_stream->empty(query);
}

rpy::algebra::Lie rpy::streams::RestrictionStream::log_signature_impl(
    const intervals::Interval& interval,
    const algebra::Context& ctx) const
{
    RPY_CHECK(m_domain.intersects_with(interval));

    const auto query = intersection(m_domain, interval);
    return p_stream->log_signature(query, ctx);
}