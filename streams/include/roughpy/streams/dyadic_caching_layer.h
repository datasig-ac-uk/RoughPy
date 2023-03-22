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
class DyadicCachingLayer : public StreamInterface {
    mutable std::map<intervals::DyadicInterval, algebra::Lie> m_cache;
    mutable std::recursive_mutex m_compute_lock;

public:
    using StreamInterface::StreamInterface;

    DyadicCachingLayer(const DyadicCachingLayer&) = delete;
    DyadicCachingLayer(DyadicCachingLayer&& other) noexcept;

    DyadicCachingLayer& operator=(const DyadicCachingLayer&) = delete;
    DyadicCachingLayer& operator=(DyadicCachingLayer&& other) noexcept;


    using StreamInterface::log_signature;

    algebra::Lie
    log_signature(const intervals::DyadicInterval &interval,
                  resolution_t resolution,
                  const algebra::Context &ctx);

    algebra::Lie
    log_signature(const intervals::Interval &domain,
                  resolution_t resolution,
                  const algebra::Context &ctx);
};

}// namespace streams
}// namespace rpy
#endif// ROUGHPY_STREAMS_DYADIC_CACHING_LAYER_H_
