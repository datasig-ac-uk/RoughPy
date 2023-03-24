#ifndef ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_
#define ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_

#include "stream_base.h"

#include "dyadic_caching_layer.h"

namespace rpy {
namespace streams {

class DynamicallyConstructedStream : public DyadicCachingLayer  {
public:

    using DyadicCachingLayer::DyadicCachingLayer;




};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_
