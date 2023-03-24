#ifndef ROUGHPY_STREAMS_TICK_STREAM_H_
#define ROUGHPY_STREAMS_TICK_STREAM_H_

#include "stream_base.h"

#include <roughpy/intervals/dyadic_interval.h>
#include <roughpy/scalars/scalar_stream.h>


#include <map>
#include <vector>

namespace rpy {
namespace streams {

class ROUGHPY_STREAMS_EXPORT TickStream : public StreamInterface {
    std::vector<param_t> m_granular_times;
    std::map<intervals::DyadicInterval, algebra::Lie> m_data;
    resolution_t m_resolution;
    intervals::IntervalType m_itype;

    using DyadicInterval = intervals::DyadicInterval;

    optional<DyadicInterval> smallest_dyadic_containing_all_events(const DyadicInterval& di, resolution_t resolution) const;
    optional<DyadicInterval> smallest_dyadic_containing_all_negative_events() const;
    optional<DyadicInterval> smallest_dyadic_containing_all_positive_events() const;

    algebra::Lie recursive_logsig(DyadicInterval di);

public:

    TickStream(scalars::ScalarStream&& raw_data,
               std::vector<const key_type*> raw_key_stream,
               std::vector<param_t> raw_timestamps,
               resolution_t resolution,
               StreamMetadata md,
               intervals::IntervalType itype=intervals::IntervalType::Clopen);

    bool empty(const intervals::Interval &interval) const noexcept override;
    algebra::Lie log_signature(const DyadicInterval &interval, resolution_t resolution, const algebra::Context &ctx) const override;
    algebra::Lie log_signature(const intervals::Interval &interval, resolution_t resolution, const algebra::Context &ctx) const override;
    algebra::FreeTensor signature(const intervals::Interval &interval, const algebra::Context &ctx) const override;
    algebra::FreeTensor signature(const intervals::Interval &interval, resolution_t resolution, const algebra::Context &ctx) const override;

protected:
    algebra::Lie log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const override;
};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_TICK_STREAM_H_
