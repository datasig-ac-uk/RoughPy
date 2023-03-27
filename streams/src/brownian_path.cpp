//
// Created by user on 24/03/23.
//
#include "brownian_stream.h"

#include <roughpy/scalars/key_scalar_array.h>

using namespace rpy;
using namespace rpy::streams;

algebra::Lie BrownianStream::gaussian_increment(const algebra::Context& ctx, param_t length) const {
    const auto& md = metadata();
    scalars::KeyScalarArray incr (p_generator->normal_random(scalars::Scalar(0.), scalars::Scalar(length), md.width));
    return ctx.construct_lie({std::move(incr), md.cached_vector_type});
}

algebra::Lie BrownianStream::log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const {
    return algebra::Lie();
}
