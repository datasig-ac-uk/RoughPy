//
// Created by sammorley on 06/12/24.
//

#include "roughpy/streams/restriction_stream.h"

#include "roughpy/intervals/interval.h"


using namespace rpy;
using namespace rpy::streams;

const std::shared_ptr<StreamMetadata>& RestrictionStream::
metadata() const noexcept
{
    return p_stream->metadata();
}

const intervals::RealInterval& RestrictionStream::support() const noexcept
{
    return m_domain;
}

StreamInterface::Lie RestrictionStream::log_signature(
    const DyadicInterval& interval,
    resolution_t resolution,
    const Context& context) const
{
    auto query = intersection(interval, m_domain);
    if (query.inf() == query.sup()) {
        return this->zero_lie();
    }
    return p_stream->log_signature(query, resolution, context);
}

StreamInterface::Lie RestrictionStream::log_signature(const Interval& interval,
    resolution_t resolution,
    const Context& context) const
{
    auto query = intersection(interval, m_domain);
    if (query.inf() == query.sup()) {
        return this->zero_lie();
    }
    return p_stream->log_signature(query, resolution, context);
}
