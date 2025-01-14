//
// Created by sammorley on 06/12/24.
//

#ifndef ROUGHPY_STREAMS_RESTRICTION_STREAM_H
#define ROUGHPY_STREAMS_RESTRICTION_STREAM_H

#include <memory>

#include "roughpy/intervals/real_interval.h"

#include "stream_base.h"
#include "roughpy_streams_export.h"

namespace rpy::streams {

class ROUGHPY_STREAMS_EXPORT RestrictionStream : public StreamInterface {
    intervals::RealInterval m_domain;
    std::shared_ptr<const StreamInterface> p_stream;


public:

    RestrictionStream(std::shared_ptr<const StreamInterface> stream, intervals::RealInterval domain)
        : m_domain(std::move(domain)), p_stream(std::move(stream))
    {}


    RPY_NO_DISCARD const std::shared_ptr<StreamMetadata>&
    metadata() const noexcept override;

    RPY_NO_DISCARD const intervals::RealInterval&
    support() const noexcept override;

    RPY_NO_DISCARD Lie log_signature(const DyadicInterval& interval,
        resolution_t resolution,
        const Context& context) const override;

    RPY_NO_DISCARD Lie log_signature(const Interval& interval,
        resolution_t resolution,
        const Context& context) const override;
};



}

#endif //ROUGHPY_STREAMS_RESTRICTION_STREAM_H
