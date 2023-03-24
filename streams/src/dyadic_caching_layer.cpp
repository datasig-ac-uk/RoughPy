//
// Created by user on 10/03/23.
//

#include "dyadic_caching_layer.h"
#include <iostream>

using namespace rpy;
using namespace rpy::streams;

DyadicCachingLayer::DyadicCachingLayer(DyadicCachingLayer &&other) noexcept
    : StreamInterface(static_cast<StreamInterface&&>(other))
{
    std::lock_guard<std::recursive_mutex> access(other.m_compute_lock);
    m_cache = std::move(other.m_cache);
}
DyadicCachingLayer &DyadicCachingLayer::operator=(DyadicCachingLayer &&other) noexcept {
    if (&other != this) {
        std::lock_guard<std::recursive_mutex> this_access(m_compute_lock);
        std::lock_guard<std::recursive_mutex> that_access(other.m_compute_lock);
        m_cache = std::move(other.m_cache);
        StreamInterface::operator=(std::move(other));
    }
    return *this;
}
algebra::Lie DyadicCachingLayer::log_signature(const intervals::DyadicInterval &interval, resolution_t resolution, const algebra::Context &ctx) const {
    if (empty(interval)) {
        return ctx.zero_lie(DyadicCachingLayer::metadata().cached_vector_type);
    }

    if (interval.power() == resolution) {
        std::lock_guard<std::recursive_mutex> access(m_compute_lock);

        auto &cached = m_cache[interval];
        if (!cached) {
            cached = log_signature_impl(interval, ctx);
        }
//        std::cerr << interval << ' ';
//        cached.print(std::cerr) << '\n';
        return cached;
    }

    intervals::DyadicInterval lhs_itvl(interval);
    intervals::DyadicInterval rhs_itvl(interval);
    lhs_itvl.shrink_interval_left();
    rhs_itvl.shrink_interval_right();

    auto lhs = log_signature(lhs_itvl, resolution, ctx);
    auto rhs = log_signature(rhs_itvl, resolution, ctx);

    if (lhs.size() == 0) {
        return rhs;
    }
    if (rhs.size() == 0) {
        return lhs;
    }

    return ctx.cbh({lhs, rhs}, DyadicCachingLayer::metadata().cached_vector_type);
}
algebra::Lie DyadicCachingLayer::log_signature(const intervals::Interval &domain, resolution_t resolution, const algebra::Context &ctx) const {
    // For now, if the ctx depth is not the same as md depth just do the calculation without caching
    // be smarter about this in the future.
    const auto &md = DyadicCachingLayer::metadata();
    assert(ctx.width() == md.width);
    if (ctx.depth() != md.default_context->depth()) {
        return StreamInterface::log_signature(domain, resolution, ctx);
    }

    auto dyadic_dissection = intervals::to_dyadic_intervals(domain, resolution);

    std::vector<algebra::Lie> lies;
    lies.reserve(dyadic_dissection.size());
    for (const auto &itvl : dyadic_dissection) {
        auto lsig = log_signature(itvl, resolution, ctx);
        std::cerr << itvl << ' ';
        lsig.print(std::cerr) << '\n';
        lies.push_back(lsig);
    }

    return ctx.cbh(lies, DyadicCachingLayer::metadata().cached_vector_type);
}
algebra::Lie DyadicCachingLayer::log_signature(const intervals::Interval &interval, const algebra::Context &ctx) const {
    return log_signature(interval, metadata().default_resolution, ctx);
}
algebra::FreeTensor DyadicCachingLayer::signature(const intervals::Interval &interval, const algebra::Context &ctx) const {
    return signature(interval, metadata().default_resolution, ctx);
}
