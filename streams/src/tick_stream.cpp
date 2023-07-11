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

//
// Created by user on 10/03/23.
//

#include <roughpy/streams/tick_stream.h>

#include <set>
#include <vector>

using namespace rpy;
optional<streams::TickStream::DyadicInterval>
streams::TickStream::smallest_dyadic_containing_all_events(
        const streams::TickStream::DyadicInterval& di,
        resolution_t resolution) const
{
    // std::lower_bound returns an iterator pointing to the first element
    // in the range [first,last) which does not compare less than val
    auto be = std::lower_bound(m_granular_times.begin(), m_granular_times.end(),
                               di.inf());
    auto en = std::lower_bound(be, m_granular_times.end(), di.sup());

    // clopen be->[t2) en->[t4)
    //         [t1)[[t2),,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,[t3))[t4)
    //            inf                                        sup
    // opencl be->(t1] en->(t3]
    //         (t1]((t2],,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,(t3]](t4]
    //            inf                                        sup

    if (en != be) {
        DyadicInterval back;
        DyadicInterval i;
        DyadicInterval j;
        if (di.type() == intervals::IntervalType::Clopen) {
            back = DyadicInterval(*(--en), resolution);
            i = DyadicInterval(*be, resolution);
        } else {
            back = DyadicInterval(*en, resolution);
            i = DyadicInterval(*(++be), resolution);
        }

        for (; !i.contains_dyadic(DyadicInterval(back.dincluded_end()));
             j = i.expand_interval(), i = j) {};
        return i;
    }

    return {};
}
optional<streams::TickStream::DyadicInterval>
streams::TickStream::smallest_dyadic_containing_all_negative_events() const
{
    auto zeu = (metadata().interval_type == intervals::IntervalType::Clopen)
            ? std::upper_bound(m_granular_times.begin(), m_granular_times.end(),
                               param_t(0))
            : std::lower_bound(m_granular_times.begin(), m_granular_times.end(),
                               param_t(0));

    if (zeu != m_granular_times.begin()) {
        DyadicInterval end(*(--zeu), m_resolution);
        DyadicInterval i(*m_granular_times.begin(), m_resolution);
        DyadicInterval j;
        for (; !i.contains_dyadic(end); j = i.expand_interval(), i = j) {};

        return i;
    }
    return {};
}
optional<streams::TickStream::DyadicInterval>
streams::TickStream::smallest_dyadic_containing_all_positive_events() const
{
    const bool zero_negative
            = (DyadicInterval(0, m_resolution, metadata().interval_type)
                       .excluded_end()
               < 0);
    auto zeu = (zero_negative)
            ? std::upper_bound(m_granular_times.begin(), m_granular_times.end(),
                               param_t(0))
            : std::lower_bound(m_granular_times.begin(), m_granular_times.end(),
                               param_t(0));

    if (zeu != m_granular_times.end()) {
        DyadicInterval end(m_granular_times.back(), m_resolution);
        DyadicInterval i(*(zeu), m_resolution);
        DyadicInterval j;
        for (; !i.contains_dyadic(end); j = i.expand_interval(), i = j) {};

        return i;
    }

    return {};
}
algebra::Lie
streams::TickStream::recursive_logsig(streams::TickStream::DyadicInterval di)
{
    const auto& md = metadata();
    const algebra::Context& ctx = *md.default_context;

    if (auto pdi1 = smallest_dyadic_containing_all_events(di, m_resolution)) {
        auto& it = m_data[*pdi1];
        if (!it.is_zero()) { return it; }

        std::vector<algebra::Lie> v;
        v.reserve(2);
        DyadicInterval left(*pdi1);
        DyadicInterval right(*pdi1);
        left.shrink_interval_left();
        right.shrink_interval_right();
        v.emplace_back(recursive_logsig(left));
        v.emplace_back(recursive_logsig(right));

        return it = ctx.cbh(v, md.cached_vector_type);
    }

    return ctx.zero_lie(md.cached_vector_type);
}
streams::TickStream::TickStream(scalars::ScalarStream&& raw_data,
                                std::vector<const key_type*> raw_key_stream,
                                std::vector<param_t> raw_timestamps,
                                resolution_t resolution, StreamMetadata md,
                                intervals::IntervalType itype)
    : StreamInterface(std::move(md)), m_resolution(resolution)
{
    {
        const auto size = raw_timestamps.size();
        const auto& smeta = metadata();
        const auto& ctx = *smeta.default_context;
        std::set<param_t> index;

        for (dimn_t i = 0; i < size; ++i) {
            const DyadicInterval di(raw_timestamps[i], m_resolution,
                                    smeta.interval_type);

            const algebra::VectorConstructionData cdata{
                    {raw_data[i], raw_key_stream[i]},
                    smeta.cached_vector_type};
            auto new_lie = ctx.construct_lie(cdata);

            auto& existing = m_data[di];
            if (existing) {
                existing = ctx.cbh(existing, new_lie, smeta.cached_vector_type);
            } else {
                existing = std::move(new_lie);
            }

            index.insert(di.included_end());
        }

        m_granular_times.assign(index.begin(), index.end());
    }

    if (auto di_negative = smallest_dyadic_containing_all_negative_events()) {
        recursive_logsig(*di_negative);
    }
    if (auto di_positive = smallest_dyadic_containing_all_positive_events()) {
        recursive_logsig(*di_positive);
    }
}

streams::TickStream::TickStream(streams::StreamConstructionHelper&& helper,
                                streams::StreamMetadata md,
                                resolution_t resolution)
    : StreamInterface(std::move(md), helper.take_schema()),
      m_resolution(resolution)
{
    std::set<param_t> index;
    const auto& ctx = *metadata().default_context;
    const auto& itype = metadata().interval_type;
    const auto& vtype = metadata().cached_vector_type;

    for (auto&& item : helper.finalise()) {
        const DyadicInterval di(item.first, m_resolution, itype);
        index.insert(di.included_end());

        auto& existing = m_data[di];
        if (existing) {
            existing = ctx.cbh(existing, item.second, vtype);
        } else {
            existing = std::move(item.second);
        }
    }

    m_granular_times.assign(index.begin(), index.end());

    if (auto di_negative = smallest_dyadic_containing_all_negative_events()) {
        recursive_logsig(*di_negative);
    }
    if (auto di_positive = smallest_dyadic_containing_all_positive_events()) {
        recursive_logsig(*di_positive);
    }
}
algebra::Lie
streams::TickStream::log_signature_impl(const intervals::Interval& interval,
                                        const algebra::Context& ctx) const
{
    RPY_DBG_ASSERT(dynamic_cast<const DyadicInterval*>(&interval) == &interval);
    if (auto dil = smallest_dyadic_containing_all_events(
                static_cast<const DyadicInterval&>(interval), m_resolution)) {
        auto found = m_data.find(*dil);
        RPY_DBG_ASSERT(found != m_data.end());
        return found->second;
    }
    return ctx.zero_lie(metadata().cached_vector_type);
}
bool streams::TickStream::empty(
        const intervals::Interval& interval) const noexcept
{
    auto dissection = intervals::to_dyadic_intervals(interval, m_resolution);

    for (auto& di : dissection) {
        if (smallest_dyadic_containing_all_events(di, m_resolution)) {
            return false;
        }
    }
    return true;
}
algebra::Lie
streams::TickStream::log_signature(const intervals::DyadicInterval& interval,
                                   resolution_t resolution,
                                   const algebra::Context& ctx) const
{
    return StreamInterface::log_signature(interval, resolution, ctx);
}
algebra::Lie
streams::TickStream::log_signature(const intervals::Interval& interval,
                                   resolution_t resolution,
                                   const algebra::Context& ctx) const
{
    return StreamInterface::log_signature(interval, resolution, ctx);
}
algebra::FreeTensor
streams::TickStream::signature(const intervals::Interval& interval,
                               const algebra::Context& ctx) const
{
    return StreamInterface::signature(interval, ctx);
}
algebra::FreeTensor
streams::TickStream::signature(const intervals::Interval& interval,
                               resolution_t resolution,
                               const algebra::Context& ctx) const
{
    return StreamInterface::signature(interval, resolution, ctx);
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::TickStream

#include <roughpy/platform/serialization_instantiations.inl>
