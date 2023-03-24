//
// Created by user on 09/03/23.
//

#include "stream_base.h"

#include <iostream>
using namespace rpy;
using namespace rpy::streams;

const StreamMetadata &StreamInterface::metadata() const noexcept {
    return m_metadata;
}
bool StreamInterface::empty(const intervals::Interval &interval) const noexcept {
    if (interval.type() == intervals::IntervalType::Clopen) {
        return interval.sup() < m_metadata.effective_support.inf()
                || interval.inf() >= m_metadata.effective_support.sup();
    }
    return interval.sup() <= m_metadata.effective_support.inf()
            || interval.inf() > m_metadata.effective_support.sup();
}

algebra::Lie StreamInterface::log_signature(const intervals::Interval &interval, const algebra::Context &ctx) const {
    return log_signature_impl(interval, ctx);
}

rpy::algebra::Lie rpy::streams::StreamInterface::log_signature(const rpy::intervals::DyadicInterval &interval, rpy::streams::resolution_t /* resolution*/, const rpy::algebra::Context &ctx) const {
    auto result = log_signature_impl(interval, ctx);
    return result;
}
rpy::algebra::Lie rpy::streams::StreamInterface::log_signature(const rpy::intervals::Interval &interval, rpy::streams::resolution_t resolution, const rpy::algebra::Context &ctx) const {
    auto dissection = intervals::to_dyadic_intervals(interval, resolution);
    std::vector<algebra::Lie> lies;
    lies.reserve(dissection.size());

    for (auto& ivl : dissection) {
        lies.push_back(log_signature_impl(ivl, ctx));
        std::cerr << ivl << ' ';
        lies.back().print(std::cerr) << '\n';
    }

    return ctx.cbh(lies, m_metadata.cached_vector_type);
}
algebra::FreeTensor StreamInterface::signature(const intervals::Interval &interval, const algebra::Context &ctx) const {
    return ctx.lie_to_tensor(log_signature_impl(interval, ctx)).exp();
}
rpy::algebra::FreeTensor rpy::streams::StreamInterface::signature(const rpy::intervals::Interval &interval, rpy::streams::resolution_t resolution, const rpy::algebra::Context &ctx) const {
    return ctx.lie_to_tensor(log_signature(interval, resolution, ctx)).exp();
}
