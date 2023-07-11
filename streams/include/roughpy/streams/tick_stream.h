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

#ifndef ROUGHPY_STREAMS_TICK_STREAM_H_
#define ROUGHPY_STREAMS_TICK_STREAM_H_

#include "stream_base.h"

#include <roughpy/intervals/dyadic_interval.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/scalar_stream.h>

#include "stream_construction_helper.h"

#include <map>
#include <vector>

namespace rpy {
namespace streams {

class RPY_EXPORT TickStream : public StreamInterface
{
    std::vector<param_t> m_granular_times;
    std::map<intervals::DyadicInterval, algebra::Lie> m_data;
    resolution_t m_resolution;

    using DyadicInterval = intervals::DyadicInterval;

    RPY_NO_DISCARD
    optional<DyadicInterval>
    smallest_dyadic_containing_all_events(const DyadicInterval& di,
                                          resolution_t resolution) const;
    RPY_NO_DISCARD
    optional<DyadicInterval>
    smallest_dyadic_containing_all_negative_events() const;
    RPY_NO_DISCARD
    optional<DyadicInterval>
    smallest_dyadic_containing_all_positive_events() const;

    algebra::Lie recursive_logsig(DyadicInterval di);

public:
    TickStream(std::vector<param_t>&& granular_times,
               std::map<intervals::DyadicInterval, algebra::Lie>&& data,
               resolution_t resolution,
               std::shared_ptr<streams::StreamSchema> schema,
               StreamMetadata&& md)
        : StreamInterface(std::move(md), std::move(schema)),
          m_granular_times(std::move(granular_times)), m_data(std::move(data)),
          m_resolution(resolution)
    {
        if (auto di_negative
            = smallest_dyadic_containing_all_negative_events()) {
            recursive_logsig(*di_negative);
        }
        if (auto di_positive
            = smallest_dyadic_containing_all_positive_events()) {
            recursive_logsig(*di_positive);
        }
    }

    TickStream(StreamConstructionHelper&& helper, StreamMetadata md,
               resolution_t resolution);

    TickStream(scalars::ScalarStream&& raw_data,
               std::vector<const key_type*> raw_key_stream,
               std::vector<param_t> raw_timestamps, resolution_t resolution,
               StreamMetadata md,
               intervals::IntervalType itype = intervals::IntervalType::Clopen);

    RPY_NO_DISCARD
    bool empty(const intervals::Interval& interval) const noexcept override;
    RPY_NO_DISCARD
    algebra::Lie log_signature(const DyadicInterval& interval,
                               resolution_t resolution,
                               const algebra::Context& ctx) const override;
    RPY_NO_DISCARD
    algebra::Lie log_signature(const intervals::Interval& interval,
                               resolution_t resolution,
                               const algebra::Context& ctx) const override;
    RPY_NO_DISCARD
    algebra::FreeTensor signature(const intervals::Interval& interval,
                                  const algebra::Context& ctx) const override;
    RPY_NO_DISCARD
    algebra::FreeTensor signature(const intervals::Interval& interval,
                                  resolution_t resolution,
                                  const algebra::Context& ctx) const override;

protected:
    RPY_NO_DISCARD
    algebra::Lie log_signature_impl(const intervals::Interval& interval,
                                    const algebra::Context& ctx) const override;

public:
    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(TickStream)
{
    auto md = metadata();
    RPY_SERIAL_SERIALIZE_NVP("metadata", md);
    RPY_SERIAL_SERIALIZE_NVP("granular_times", m_granular_times);
    RPY_SERIAL_SERIALIZE_NVP("data", m_data);
    RPY_SERIAL_SERIALIZE_NVP("resolution", m_resolution);
}

}// namespace streams
}// namespace rpy

#ifndef RPY_DISABLE_SERIALIZATION
RPY_SERIAL_LOAD_AND_CONSTRUCT(rpy::streams::TickStream)
{
    using namespace rpy;
    using namespace rpy::streams;

    StreamMetadata metadata;
    RPY_SERIAL_SERIALIZE_VAL(metadata);
    std::vector<param_t> granular_times;
    RPY_SERIAL_SERIALIZE_VAL(granular_times);
    std::map<intervals::DyadicInterval, algebra::Lie> data;
    RPY_SERIAL_SERIALIZE_VAL(data);
    resolution_t resolution;
    RPY_SERIAL_SERIALIZE_VAL(resolution);
    intervals::IntervalType interval_type;
    RPY_SERIAL_SERIALIZE_VAL(interval_type);

    construct(std::move(granular_times), std::move(data), resolution,
              interval_type, std::move(metadata));
}

#endif

#endif// ROUGHPY_STREAMS_TICK_STREAM_H_
