#ifndef ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_
#define ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_

#include "stream_base.h"

#include "dyadic_caching_layer.h"

namespace rpy {
namespace streams {

class DynamicallyConstructedStream : public DyadicCachingLayer  {
public:

    using DyadicCachingLayer::DyadicCachingLayer;

    algebra::Lie log_signature(const intervals::DyadicInterval &interval, resolution_t resolution, const algebra::Context &ctx) const override;
    algebra::Lie log_signature(const intervals::Interval &domain, resolution_t resolution, const algebra::Context &ctx) const override;
};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_
