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

#include <roughpy/streams/dyadic_caching_layer.h>

using namespace rpy;
using namespace rpy::streams;

DyadicCachingLayer::DyadicCachingLayer(DyadicCachingLayer&& other) noexcept
    : StreamInterface(static_cast<StreamInterface&&>(other))
{
    std::lock_guard<std::recursive_mutex> access(other.m_compute_lock);
    m_cache = std::move(other.m_cache);
}
DyadicCachingLayer&
DyadicCachingLayer::operator=(DyadicCachingLayer&& other) noexcept
{
    if (&other != this) {
        std::lock_guard<std::recursive_mutex> this_access(m_compute_lock);
        std::lock_guard<std::recursive_mutex> that_access(other.m_compute_lock);
        m_cache = std::move(other.m_cache);
        StreamInterface::operator=(std::move(other));
    }
    return *this;
}
algebra::Lie
DyadicCachingLayer::log_signature(const intervals::DyadicInterval& interval,
                                  resolution_t resolution,
                                  const algebra::Context& ctx) const
{
    if (empty(interval)) {
        return ctx.zero_lie(DyadicCachingLayer::metadata().cached_vector_type);
    }

    if (interval.power() == resolution) {
        std::lock_guard<std::recursive_mutex> access(m_compute_lock);

        auto& cached = m_cache[interval];
        if (!cached) { cached = log_signature_impl(interval, ctx); }
        // Currently, const borrowing is not permitted, so return a mutable
        // view.
        return cached.borrow_mut();
    }

    intervals::DyadicInterval lhs_itvl(interval);
    intervals::DyadicInterval rhs_itvl(interval);
    lhs_itvl.shrink_interval_left();
    rhs_itvl.shrink_interval_right();

    auto lhs = log_signature(lhs_itvl, resolution, ctx);
    auto rhs = log_signature(rhs_itvl, resolution, ctx);

    if (lhs.size() == 0) { return rhs; }
    if (rhs.size() == 0) { return lhs; }

    return ctx.cbh({lhs, rhs},
                   DyadicCachingLayer::metadata().cached_vector_type);
}
algebra::Lie
DyadicCachingLayer::log_signature(const intervals::Interval& domain,
                                  resolution_t resolution,
                                  const algebra::Context& ctx) const
{
    // For now, if the ctx depth is not the same as md depth just do the
    // calculation without caching be smarter about this in the future.
    const auto& md = DyadicCachingLayer::metadata();
    RPY_CHECK(ctx.width() == md.width);
    if (ctx.depth() != md.default_context->depth()) {
        return StreamInterface::log_signature(domain, resolution, ctx);
    }

    auto dyadic_dissection = intervals::to_dyadic_intervals(domain, resolution);

    std::vector<algebra::Lie> lies;
    lies.reserve(dyadic_dissection.size());
    for (const auto& itvl : dyadic_dissection) {
        auto lsig = log_signature(itvl, resolution, ctx);
        lies.push_back(lsig);
    }

    return ctx.cbh(lies, DyadicCachingLayer::metadata().cached_vector_type);
}
algebra::Lie
DyadicCachingLayer::log_signature(const intervals::Interval& interval,
                                  const algebra::Context& ctx) const
{
    return log_signature(interval, metadata().default_resolution, ctx);
}
algebra::FreeTensor
DyadicCachingLayer::signature(const intervals::Interval& interval,
                              const algebra::Context& ctx) const
{
    return signature(interval, metadata().default_resolution, ctx);
}
