//
// Created by sammorley on 03/12/24.
//

#ifndef ROUGHPY_STREAMS_VALUE_STREAM_H
#define ROUGHPY_STREAMS_VALUE_STREAM_H

#include <memory>

#include <roughpy/core/macros.h>

#include "arrival_stream.h"

namespace rpy {
namespace streams {

class ValueStream : public StreamInterface
{
    intervals::RealInterval m_domain;
    std::shared_ptr<const ArrivalStream> p_arrival_stream;
    std::shared_ptr<const StreamInterface> p_increment_stream;

public:
    ValueStream(
        intervals::RealInterval domain,
        std::shared_ptr<const ArrivalStream> arrival_stream,
        std::shared_ptr<const StreamInterface> increment_stream)
        : m_domain(domain),
          p_arrival_stream(std::move(arrival_stream)),
          p_increment_stream(std::move(increment_stream)) {}

    RPY_NO_DISCARD
    std::shared_ptr<const ValueStream> query(
        const intervals::Interval& interval) const;

    RPY_NO_DISCARD bool
    empty(const intervals::Interval& interval) const noexcept override;

protected:
    RPY_NO_DISCARD algebra::Lie log_signature_impl(
        const intervals::Interval& interval,
        const algebra::Context& ctx) const override;
};

}// streams
}// rpy

#endif //ROUGHPY_STREAMS_VALUE_STREAM_H