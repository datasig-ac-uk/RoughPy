//
// Created by user on 09/03/23.
//

#include "stream_base.h"

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
rpy::algebra::Lie rpy::streams::StreamInterface::log_signature(const rpy::intervals::DyadicInterval &interval, rpy::streams::resolution_t resolution, const rpy::algebra::Context &ctx) const {
    return log_signature(interval, ctx);
}
rpy::algebra::Lie rpy::streams::StreamInterface::log_signature(const rpy::intervals::Interval &interval, rpy::streams::resolution_t resolution, const rpy::algebra::Context &ctx) const {
    return log_signature(interval, ctx);
}
rpy::algebra::FreeTensor rpy::streams::StreamInterface::signature(const rpy::intervals::Interval &interval, rpy::streams::resolution_t resolution, const rpy::algebra::Context &ctx) const {
    return ctx.lie_to_tensor(log_signature(interval, ctx)).exp();
}
