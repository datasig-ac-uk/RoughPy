// Copyright (c) 2023 RoughPy Developers. All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_STREAMS_BROWNIAN_STREAM_H_
#define ROUGHPY_STREAMS_BROWNIAN_STREAM_H_

#include "stream_base.h"
#include "dynamically_constructed_stream.h"
#include <roughpy/scalars/random.h>
#include <roughpy/platform/serialization.h>

#include <memory>

namespace rpy {
namespace streams {

class ROUGHPY_STREAMS_EXPORT BrownianStream : public DynamicallyConstructedStream {
    std::unique_ptr<scalars::RandomGenerator> p_generator;

    algebra::Lie gaussian_increment(const algebra::Context& ctx, param_t length) const;
protected:
    algebra::Lie log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const override;
    Lie make_new_root_increment(DyadicInterval di) const override;
    Lie make_neighbour_root_increment(DyadicInterval neighbour_di) const override;
    pair<Lie, Lie> compute_child_lie_increments(DyadicInterval left_di, DyadicInterval right_di, const Lie &parent_value) const override;

public:

    scalars::RandomGenerator& generator() noexcept { return *p_generator; }

    BrownianStream(std::unique_ptr<scalars::RandomGenerator> generator, StreamMetadata md)
        : DynamicallyConstructedStream(std::move(md)), p_generator(std::move(generator))
    {}


    RPY_SERIAL_SERIALIZE_FN();

};

RPY_SERIAL_SERIALIZE_FN_IMPL(BrownianStream) {
    auto md = metadata();
    RPY_SERIAL_SERIALIZE_NVP("metadata", md);
    std::string generator = p_generator->get_type();
    RPY_SERIAL_SERIALIZE_NVP("generator", generator);
    auto state = p_generator->get_state();
    RPY_SERIAL_SERIALIZE_NVP("state", state);
}

}
}// namespace rpy

RPY_SERIAL_LOAD_AND_CONSTRUCT(rpy::streams::BrownianStream) {
    using namespace rpy;
    using namespace rpy::streams;

    StreamMetadata md;
    RPY_SERIAL_SERIALIZE_NVP("metadata", md);

    std::string generator;
    RPY_SERIAL_SERIALIZE_VAL(generator);

    std::string state;
    RPY_SERIAL_SERIALIZE_VAL(state);
    const auto *stype = md.data_scalar_type;

    construct(stype->get_rng(generator), std::move(md));

    construct->generator().set_state(state);
}


RPY_SERIAL_REGISTER_CLASS(rpy::streams::BrownianStream)

#endif// ROUGHPY_STREAMS_BROWNIAN_STREAM_H_
