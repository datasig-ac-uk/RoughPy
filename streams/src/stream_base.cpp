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
// Created by user on 09/03/23.
//

#include <roughpy/streams/stream_base.h>

using namespace rpy;
using namespace rpy::streams;

void StreamInterface::set_metadata(StreamMetadata&& md) noexcept
{
    m_metadata = std::move(md);
}

StreamInterface::~StreamInterface() noexcept = default;

bool StreamInterface::empty(const intervals::Interval& interval) const noexcept
{
    if (interval.type() == intervals::IntervalType::Clopen) {
        return interval.sup() < m_metadata.effective_support.inf()
                || interval.inf() >= m_metadata.effective_support.sup();
    }
    return interval.sup() <= m_metadata.effective_support.inf()
            || interval.inf() > m_metadata.effective_support.sup();
}

algebra::Lie StreamInterface::log_signature(const intervals::Interval& interval,
                                            const algebra::Context& ctx) const
{
    return log_signature_impl(interval, ctx);
}

rpy::algebra::Lie rpy::streams::StreamInterface::log_signature(
        const rpy::intervals::DyadicInterval& interval,
        rpy::resolution_t /* resolution*/,
        const rpy::algebra::Context& ctx) const
{
    auto result = log_signature_impl(interval, ctx);
    return result;
}
rpy::algebra::Lie rpy::streams::StreamInterface::log_signature(
        const rpy::intervals::Interval& interval, rpy::resolution_t resolution,
        const rpy::algebra::Context& ctx) const
{
    auto dissection = intervals::to_dyadic_intervals(interval, resolution);
    std::vector<algebra::Lie> lies;
    lies.reserve(dissection.size());

    for (auto& ivl : dissection) {
        lies.push_back(log_signature_impl(ivl, ctx));
    }

    return ctx.cbh(lies, m_metadata.cached_vector_type);
}
algebra::FreeTensor
StreamInterface::signature(const intervals::Interval& interval,
                           const algebra::Context& ctx) const
{
    return ctx.lie_to_tensor(log_signature_impl(interval, ctx)).exp();
}
rpy::algebra::FreeTensor rpy::streams::StreamInterface::signature(
        const rpy::intervals::Interval& interval, rpy::resolution_t resolution,
        const rpy::algebra::Context& ctx) const
{
    return ctx.lie_to_tensor(log_signature(interval, resolution, ctx)).exp();
}

//
//
//
// #define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::StreamInterface
// #define RPY_SERIAL_DO_SPLIT
// #include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME StreamMetadata
#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>
