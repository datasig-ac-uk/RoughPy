//
// Created by user on 18/03/23.
//

#include "dynamically_constructed_stream.h"

using namespace rpy;
using namespace rpy::streams;

algebra::Lie DynamicallyConstructedStream::log_signature(const intervals::DyadicInterval &interval, resolution_t resolution, const algebra::Context &ctx) const {
    const auto& md = metadata();
    if (!interval.intersects_with(md.effective_support)) {
        return ctx.zero_lie(md.cached_vector_type);
    }
    // TODO: Logic for truncating
    return DyadicCachingLayer::log_signature(interval, resolution, ctx);
}
algebra::Lie DynamicallyConstructedStream::log_signature(const intervals::Interval &domain, resolution_t resolution, const algebra::Context &ctx) const {
    const auto& md = metadata();
    if (!domain.intersects_with(md.effective_support)) {
        return ctx.zero_lie(md.cached_vector_type);
    }
    // TODO: logic for truncating
    return DyadicCachingLayer::log_signature(domain, resolution, ctx);
}
