#ifndef ROUGHPY_STREAMS_DYADIC_CACHING_LAYER_H_
#define ROUGHPY_STREAMS_DYADIC_CACHING_LAYER_H_

#include <map>
#include <mutex>

#include <roughpy/algebra/lie.h>
#include <roughpy/intervals/dyadic_interval.h>

#include "stream_base.h"

namespace rpy {
namespace streams {

/**
 * @brief Caching layer utilising a dyadic dissection of the parameter interval.
 *
 * This layer introducing caching for the computation of log signatures by
 * utilising the fact that the signature of a concatenation of paths is the
 * product of signatures (or applying the Campbell-Baker-Hausdorff formula to
 * log signatures). The parameter interval is dissected into dyadic intervals
 * of a resolution and the log signature is computed on all those dyadic intervals
 * that are contained within the requested interval. These are then combined
 * using the Campbell-Baker-Hausdorff formula to give the log signature over the
 * whole interval.
 *
 */
template <typename BaseInterface = StreamInterface>
class DyadicCachingLayer : public BaseInterface {
    mutable std::map<intervals::DyadicInterval, algebra::Lie> m_cache;
    mutable std::recursive_mutex m_compute_lock;

public:
    using BaseInterface::BaseInterface;

    DyadicCachingLayer(const DyadicCachingLayer&) = delete;
    DyadicCachingLayer(DyadicCachingLayer&& other) noexcept;

    DyadicCachingLayer& operator=(const DyadicCachingLayer&) = delete;
    DyadicCachingLayer& operator=(DyadicCachingLayer&& other) noexcept;


    using BaseInterface::log_signature;

    algebra::Lie
    log_signature(const intervals::DyadicInterval &interval,
                  resolution_t resolution,
                  const algebra::Context &ctx);

    algebra::Lie
    log_signature(const intervals::Interval &domain,
                  resolution_t resolution,
                  const algebra::Context &ctx);
};

template <typename BaseInterface>
DyadicCachingLayer<BaseInterface>::DyadicCachingLayer(DyadicCachingLayer &&other) noexcept
    : BaseInterface(static_cast<BaseInterface&&>(other))
{
    std::lock_guard<std::recursive_mutex> access(other.m_compute_lock);
    m_cache = std::move(other.m_cache);
}
template <typename BaseInterface>
DyadicCachingLayer<BaseInterface> &DyadicCachingLayer<BaseInterface>::operator=(DyadicCachingLayer &&other) noexcept {
    if (&other != this) {
        std::lock_guard<std::recursive_mutex> this_access(m_compute_lock);
        std::lock_guard<std::recursive_mutex> that_access(other.m_compute_lock);
        m_cache = std::move(other.m_cache);
        BaseInterface::operator=(std::move(other));
    }
    return *this;
}

template <typename BaseInterface>
algebra::Lie DyadicCachingLayer<BaseInterface>::log_signature(const intervals::DyadicInterval &interval, resolution_t resolution, const algebra::Context &ctx) {
    if (DyadicCachingLayer::empty(interval)) {
        return ctx.zero_lie(DyadicCachingLayer::metadata().cached_vector_type);
    }

    if (interval.power() == resolution) {
        std::lock_guard<std::recursive_mutex> access(m_compute_lock);

        auto &cached = m_cache[interval];
        if (!cached) {
            cached = log_signature(interval, ctx);
        }
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
template <typename BaseInterface>
algebra::Lie DyadicCachingLayer<BaseInterface>::log_signature(const intervals::Interval &domain, resolution_t resolution, const algebra::Context &ctx) {
    // For now, if the ctx depth is not the same as md depth just do the calculation without caching
    // be smarter about this in the future.
    const auto &md = DyadicCachingLayer::metadata();
    assert(ctx.width() == md.width);
    if (ctx.depth() != md.default_context->depth()) {
        return BaseInterface::log_signature(domain, resolution, ctx);
    }

    auto dyadic_dissection = intervals::to_dyadic_intervals(domain, resolution);

    std::vector<algebra::Lie> lies;
    lies.reserve(dyadic_dissection.size());
    for (const auto &itvl : dyadic_dissection) {
        auto lsig = log_signature(itvl, resolution, ctx);
        lies.push_back(lsig);
    }

    return ctx.cbh(lies, DyadicCachingLayer::metadata().cached_vector_type);
}

extern template class DyadicCachingLayer<StreamInterface>;
extern template class DyadicCachingLayer<SolutionStreamInterface>;

}// namespace streams
}// namespace rpy
#endif// ROUGHPY_STREAMS_DYADIC_CACHING_LAYER_H_
