// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_
#define ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_

#include "dyadic_caching_layer.h"
#include "stream_base.h"

#include <boost/container/flat_map.hpp>

#include <roughpy/core/helpers.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/key_scalar_stream.h>

namespace rpy {
namespace streams {

class ROUGHPY_STREAMS_EXPORT LieIncrementStream : public DyadicCachingLayer
{
    using base_t = DyadicCachingLayer;

public:
    using Lie = algebra::Lie;

private:
    boost::container::flat_map<param_t, Lie> m_data;

public:
    using DyadicCachingLayer::DyadicCachingLayer;

    LieIncrementStream(
            const scalars::KeyScalarArray& buffer, Slice<param_t> indices,
            StreamMetadata md, std::shared_ptr<StreamSchema> schema
    );

    explicit LieIncrementStream(
            const scalars::KeyScalarStream& ks_stream, Slice<param_t> indices,
            StreamMetadata md, std::shared_ptr<StreamSchema> schema
            );

    RPY_NO_DISCARD bool
    empty(const intervals::Interval& interval) const noexcept override;

protected:
    RPY_NO_DISCARD algebra::Lie log_signature_impl(
            const intervals::Interval& interval, const algebra::Context& ctx
    ) const override;

public:
    RPY_SERIAL_SERIALIZE_FN();
};

#ifdef RPY_COMPILING_STREAMS
RPY_SERIAL_EXTERN_SERIALIZE_CLS_BUILD(LieIncrementStream)
#else
RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMP(LieIncrementStream)
#endif

RPY_SERIAL_SERIALIZE_FN_IMPL(LieIncrementStream)
{
    RPY_SERIAL_SERIALIZE_BASE(DyadicCachingLayer);
    RPY_SERIAL_SERIALIZE_NVP("data", m_data);
}

}// namespace streams
}// namespace rpy

#ifndef RPY_DISABLE_SERIALIZATION

//RPY_SERIAL_LOAD_AND_CONSTRUCT(rpy::streams::LieIncrementStream)
//{
//    using namespace rpy;
//    using namespace rpy::streams;
//
//    StreamMetadata md;
//    RPY_SERIAL_SERIALIZE_NVP("metadata", md);
//    boost::container::flat_map<param_t, algebra::Lie> data;
//    RPY_SERIAL_SERIALIZE_VAL(data);
//
//    construct(std::move(md));
//}
//
#endif

// RPY_SERIAL_FORCE_DYNAMIC_INIT(lie_increment_stream)

RPY_SERIAL_SPECIALIZE_TYPES(rpy::streams::LieIncrementStream,
                            rpy::serial::specialization::member_serialize)

#endif// ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_
