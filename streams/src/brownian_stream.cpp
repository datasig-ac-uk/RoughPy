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
// Created by user on 24/03/23.
//
#include <roughpy/streams/brownian_stream.h>

#include <roughpy/algebra/lie.h>
#include <roughpy/scalars/key_scalar_array.h>

using namespace rpy;
using namespace rpy::streams;

algebra::Lie BrownianStream::gaussian_increment(const algebra::Context& ctx,
                                                param_t length) const
{
    const auto& md = metadata();
    scalars::KeyScalarArray incr(p_generator->normal_random(
            scalars::Scalar(0.), scalars::Scalar(length), md.width));
    return ctx.construct_lie({std::move(incr), md.cached_vector_type});
}

algebra::Lie
BrownianStream::log_signature_impl(const intervals::Interval& interval,
                                   const algebra::Context& ctx) const
{
    return algebra::Lie();
}
pair<algebra::Lie, algebra::Lie> BrownianStream::compute_child_lie_increments(
        DynamicallyConstructedStream::DyadicInterval left_di,
        DynamicallyConstructedStream::DyadicInterval right_di,
        const DynamicallyConstructedStream::Lie& parent_value) const
{
    const auto& md = metadata();
    const auto mean = parent_value.smul(md.data_scalar_type->from(1, 2));

    auto length = ldexp(0.5, left_di.power());

    const auto perturbation = gaussian_increment(*md.default_context, length);
    return {mean.add(perturbation), mean.sub(perturbation)};
}
DynamicallyConstructedStream::Lie BrownianStream::make_new_root_increment(
        DynamicallyConstructedStream::DyadicInterval di) const
{
    return gaussian_increment(*metadata().default_context, di.sup() - di.inf());
}
DynamicallyConstructedStream::Lie BrownianStream::make_neighbour_root_increment(
        DynamicallyConstructedStream::DyadicInterval neighbour_di) const
{
    return gaussian_increment(*metadata().default_context,
                              neighbour_di.sup() - neighbour_di.inf());
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::BrownianStream
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>
