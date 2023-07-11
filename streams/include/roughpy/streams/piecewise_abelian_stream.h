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

#ifndef ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_
#define ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_

#include "stream_base.h"

#include <roughpy/algebra/context.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

namespace rpy {
namespace streams {

class PiecewiseAbelianStream : public StreamInterface
{
public:
    using LiePiece = std::pair<intervals::RealInterval, algebra::Lie>;

private:
    std::vector<LiePiece> m_data;

    RPY_NO_DISCARD
    static inline scalars::Scalar
    to_multiplier_upper(const intervals::RealInterval& interval, param_t param)
    {
        RPY_DBG_ASSERT(interval.inf() <= param && param <= interval.sup());
        return scalars::Scalar((interval.sup() - param)
                               / (interval.sup() - interval.inf()));
    }
    RPY_NO_DISCARD
    static inline scalars::Scalar
    to_multiplier_lower(const intervals::RealInterval& interval, param_t param)
    {
        RPY_DBG_ASSERT(interval.inf() <= param && param <= interval.sup());
        return scalars::Scalar((param - interval.inf())
                               / (interval.sup() - interval.inf()));
    }

public:
    PiecewiseAbelianStream(std::vector<LiePiece>&& arg, StreamMetadata&& md);
    PiecewiseAbelianStream(std::vector<LiePiece>&& arg, StreamMetadata&& md,
                           std::shared_ptr<StreamSchema> schema);

    RPY_NO_DISCARD
    bool empty(const intervals::Interval& interval) const noexcept override;

protected:
    RPY_NO_DISCARD
    algebra::Lie log_signature_impl(const intervals::Interval& domain,
                                    const algebra::Context& ctx) const override;

public:
    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(PiecewiseAbelianStream)
{
    RPY_SERIAL_SERIALIZE_NVP("metadata", metadata());
    RPY_SERIAL_SERIALIZE_NVP("data", m_data);
}

}// namespace streams
}// namespace rpy

#ifndef RPY_DISABLE_SERIALIZATION
RPY_SERIAL_LOAD_AND_CONSTRUCT(rpy::streams::PiecewiseAbelianStream)
{
    using namespace rpy;
    using namespace rpy::streams;
    using LiePiece = typename PiecewiseAbelianStream::LiePiece;

    StreamMetadata metadata;
    RPY_SERIAL_SERIALIZE_VAL(metadata);
    std::vector<LiePiece> data;
    RPY_SERIAL_SERIALIZE_VAL(data);

    construct(std::move(data), std::move(metadata));
}

#endif

#endif// ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_
