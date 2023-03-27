#ifndef ROUGHPY_STREAMS_BROWNIAN_STREAM_H_
#define ROUGHPY_STREAMS_BROWNIAN_STREAM_H_

#include "stream_base.h"
#include "dynamically_constructed_stream.h"
#include <roughpy/scalars/random.h>

namespace rpy {
namespace streams {

class ROUGHPY_STREAMS_EXPORT BrownianStream : public DynamicallyConstructedStream {
    const scalars::RandomGenerator* p_generator;

    algebra::Lie gaussian_increment(const algebra::Context& ctx, param_t length) const;
protected:
    algebra::Lie log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const override;

public:


};


}
}// namespace rpy

#endif// ROUGHPY_STREAMS_BROWNIAN_STREAM_H_
