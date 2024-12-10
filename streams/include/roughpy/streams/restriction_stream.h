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
    RPY_NO_DISCARD bool
    empty(const intervals::Interval& interval) const noexcept override;

protected:
    RPY_NO_DISCARD algebra::Lie log_signature_impl(
        const intervals::Interval& interval,
        const algebra::Context& ctx) const override;


};



}

#endif //ROUGHPY_STREAMS_RESTRICTION_STREAM_H
