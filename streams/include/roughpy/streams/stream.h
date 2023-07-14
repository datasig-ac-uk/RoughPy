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

#ifndef ROUGHPY_STREAMS_STREAM_H_
#define ROUGHPY_STREAMS_STREAM_H_

#include "stream_base.h"

#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/partition.h>
#include <roughpy/platform/serialization.h>

#include <memory>

namespace rpy {
namespace streams {

class RPY_EXPORT Stream
{
public:
    using FreeTensor = algebra::FreeTensor;
    using Lie = algebra::Lie;
    using Context = algebra::Context;
    using Interval = intervals::Interval;
    using RealInterval = intervals::RealInterval;

    using perturbation_t = std::pair<RealInterval, Lie>;
    using perturbation_list_t = std::vector<perturbation_t>;

private:
    std::shared_ptr<const StreamInterface> p_impl;
    RealInterval m_support;

    RPY_NO_DISCARD FreeTensor unit_tensor(const Context& ctx) const
    {
        auto result = ctx.zero_free_tensor(metadata().cached_vector_type);
        result[0] = scalars::Scalar(1);
        return result;
    }

    RPY_NO_DISCARD Lie zero_lie(const Context& ctx) const
    {
        return ctx.zero_lie(metadata().cached_vector_type);
    }

    RPY_NO_DISCARD bool check_support_and_trim(RealInterval& domain
    ) const noexcept;

    Stream(const std::shared_ptr<const StreamInterface>& impl,
           RealInterval support)
        : p_impl(impl), m_support(support)
    {}

public:
    template <typename Impl>
    explicit Stream(Impl&& impl);

    void restrict_to(const Interval& interval);

    Stream restrict(const Interval& interval) const;

    RPY_NO_DISCARD const RealInterval& support() const noexcept
    {
        return m_support;
    }

    RPY_NO_DISCARD const StreamMetadata& metadata() const;

    RPY_NO_DISCARD const Context& get_default_context() const;

    RPY_NO_DISCARD const StreamSchema& schema() const;

    RPY_NO_DISCARD Lie log_signature() const;
    RPY_NO_DISCARD Lie log_signature(const Context& ctx) const;
    RPY_NO_DISCARD Lie log_signature(resolution_t resolution);
    RPY_NO_DISCARD Lie
    log_signature(resolution_t resolution, const Context& ctx) const;
    RPY_NO_DISCARD Lie log_signature(const Interval& interval) const;
    RPY_NO_DISCARD Lie
    log_signature(const Interval& interval, resolution_t resolution) const;
    RPY_NO_DISCARD Lie log_signature(
            const Interval& interval, resolution_t resolution,
            const Context& ctx
    ) const;

    RPY_NO_DISCARD FreeTensor signature() const;
    RPY_NO_DISCARD FreeTensor signature(const Context& ctx) const;
    RPY_NO_DISCARD FreeTensor signature(resolution_t resolution);
    RPY_NO_DISCARD FreeTensor
    signature(resolution_t resolution, const Context& ctx) const;
    RPY_NO_DISCARD FreeTensor signature(const Interval& interval) const;
    RPY_NO_DISCARD FreeTensor
    signature(const Interval& interval, resolution_t resolution) const;
    RPY_NO_DISCARD FreeTensor signature(
            const Interval& interval, resolution_t resolution,
            const Context& ctx
    ) const;

    RPY_NO_DISCARD FreeTensor
    signature_derivative(const Interval& domain, const Lie& perturbation) const;
    RPY_NO_DISCARD FreeTensor signature_derivative(
            const Interval& domain, const Lie& perturbation, const Context& ctx
    ) const;
    RPY_NO_DISCARD FreeTensor signature_derivative(
            const Interval& domain, const Lie& perturbation,
            resolution_t resolution
    ) const;
    RPY_NO_DISCARD FreeTensor signature_derivative(
            const Interval& domain, const Lie& perturbation,
            resolution_t resolution, const Context& ctx
    ) const;
    RPY_NO_DISCARD FreeTensor signature_derivative(
            const perturbation_list_t& perturbations, resolution_t resolution
    ) const;
    RPY_NO_DISCARD FreeTensor signature_derivative(
            const perturbation_list_t& perturbations, resolution_t resolution,
            const Context& ctx
    ) const;

    Stream simplify(
            const intervals::Partition& partition, resolution_t resolution
    ) const
    {
        const auto& md = metadata();
        return simplify(partition, resolution, *md.default_context);
    }

    Stream simplify(
            const intervals::Partition& partition, resolution_t resolution,
            const Context& ctx
    ) const;

    RPY_SERIAL_SERIALIZE_FN();
};

template <typename Impl>
Stream::Stream(Impl&& impl)
    : p_impl(new remove_cv_t<Impl>(std::forward<Impl>(impl))),
      m_support(p_impl->metadata().effective_support)
{}

RPY_SERIAL_SERIALIZE_FN_IMPL(Stream)
{
    RPY_SERIAL_SERIALIZE_NVP("impl", p_impl);
    RPY_SERIAL_SERIALIZE_NVP("support", m_support);
}

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_STREAM_H_
