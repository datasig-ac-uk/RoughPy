//
// Created by sammorley on 06/12/24.
//

#include "roughpy/streams/restriction_stream.h"

#include <utility>

using namespace rpy;
using namespace rpy::streams;


namespace {

intervals::RealInterval intersection(const intervals::Interval& a,
                                     const intervals::Interval& b)
{
    // TODO: Delete this once the proper function is merged
    return {std::max(a.inf(), b.inf()), std::min(a.sup(), b.sup()), a.type()};
}

}

bool rpy::streams::RestrictionStream::empty(
    const intervals::Interval& interval) const noexcept
{
    if (!m_domain.intersects_with(interval)) { return true; }
    auto query = intersection(m_domain, interval);
    return p_stream->empty(query);
}

rpy::algebra::Lie rpy::streams::RestrictionStream::log_signature_impl(
    const intervals::Interval& interval,
    const algebra::Context& ctx) const
{
    RPY_CHECK(m_domain.intersects_with(interval));

    auto query = intersection(m_domain, interval);
    return p_stream->log_signature(query, ctx);
}