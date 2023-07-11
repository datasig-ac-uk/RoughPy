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

#ifndef ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_
#define ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_

#include "dyadic_caching_layer.h"
#include "stream_base.h"

#include <boost/container/flat_map.hpp>

#include <roughpy/core/helpers.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/key_scalar_array.h>

namespace rpy {
namespace streams {

class RPY_EXPORT LieIncrementStream : public DyadicCachingLayer
{
    scalars::KeyScalarArray m_buffer;
    boost::container::flat_map<param_t, dimn_t> m_mapping;

    using base_t = DyadicCachingLayer;

public:
    LieIncrementStream(scalars::KeyScalarArray&& buffer,
                       boost::container::flat_map<param_t, dimn_t>&& mapping,
                       StreamMetadata&& md)
        : DyadicCachingLayer(std::move(md)), m_buffer(std::move(buffer)),
          m_mapping(std::move(mapping))
    {}

    LieIncrementStream(scalars::KeyScalarArray&& buffer, Slice<param_t> indices,
                       StreamMetadata md);

    RPY_NO_DISCARD
    bool empty(const intervals::Interval& interval) const noexcept override;

protected:
    RPY_NO_DISCARD
    algebra::Lie log_signature_impl(const intervals::Interval& interval,
                                    const algebra::Context& ctx) const override;

public:
    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(LieIncrementStream)
{
    StreamMetadata md = metadata();
    RPY_SERIAL_SERIALIZE_NVP("metadata", md);
    RPY_SERIAL_SERIALIZE_NVP("buffer", m_buffer);
    RPY_SERIAL_SERIALIZE_NVP("mapping", m_mapping);
}

}// namespace streams
}// namespace rpy

#ifndef RPY_DISABLE_SERIALIZATION

RPY_SERIAL_LOAD_AND_CONSTRUCT(rpy::streams::LieIncrementStream)
{
    using namespace rpy;
    using namespace rpy::streams;

    StreamMetadata md;
    RPY_SERIAL_SERIALIZE_NVP("metadata", md);
    scalars::KeyScalarArray buffer;
    RPY_SERIAL_SERIALIZE_VAL(buffer);
    boost::container::flat_map<param_t, dimn_t> mapping;
    RPY_SERIAL_SERIALIZE_VAL(mapping);

    construct(std::move(buffer), std::move(mapping), std::move(md));
}

#endif
RPY_SERIAL_REGISTER_CLASS(rpy::streams::LieIncrementStream)

#endif// ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_
