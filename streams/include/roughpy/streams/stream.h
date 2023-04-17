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

#ifndef ROUGHPY_STREAMS_STREAM_H_
#define ROUGHPY_STREAMS_STREAM_H_

#include "roughpy_streams_export.h"
#include "stream_base.h"

#include <memory>

namespace rpy {
namespace streams {

class ROUGHPY_STREAMS_EXPORT Stream {
    std::unique_ptr<const StreamInterface> p_impl;


public:
    using FreeTensor = algebra::FreeTensor;
    using Lie = algebra::Lie;
    using Context = algebra::Context;
    using Interval = intervals::Interval;
    using RealInterval = intervals::RealInterval;

    using perturbation_t = std::pair<RealInterval, Lie>;
    using perturbation_list_t = std::vector<perturbation_t>;

    template <typename Impl>
    explicit Stream(Impl &&impl);

    const StreamMetadata &metadata() const noexcept;
    const Context &get_default_context() const;

    Lie log_signature() const;
    Lie log_signature(const Context &ctx) const;
    Lie log_signature(resolution_t resolution);
    Lie log_signature(resolution_t resolution,
                      const Context &ctx) const;
    Lie log_signature(const Interval &interval) const;
    Lie log_signature(const Interval &interval,
                      resolution_t resolution) const;
    Lie log_signature(const Interval &interval,
                      resolution_t resolution,
                      const Context &ctx) const;

    FreeTensor signature() const;
    FreeTensor signature(const Context &ctx) const;
    FreeTensor signature(resolution_t resolution);
    FreeTensor signature(resolution_t resolution,
                         const Context &ctx) const;
    FreeTensor signature(const Interval &interval) const;
    FreeTensor signature(const Interval &interval,
                         resolution_t resolution) const;
    FreeTensor signature(const Interval &interval,
                         resolution_t resolution,
                         const Context &ctx) const;

    FreeTensor signature_derivative(const Interval &domain,
                                    const Lie &perturbation) const;
    FreeTensor signature_derivative(const Interval &domain,
                                    const Lie &perturbation,
                                    const Context &ctx) const;
    FreeTensor signature_derivative(const Interval &domain,
                                    const Lie &perturbation,
                                    resolution_t resolution) const;
    FreeTensor signature_derivative(const Interval &domain,
                                    const Lie &perturbation,
                                    resolution_t resolution,
                                    const Context &ctx) const;
    FreeTensor signature_derivative(const perturbation_list_t &perturbations,
                                    resolution_t resolution) const;
    FreeTensor signature_derivative(const perturbation_list_t &perturbations,
                                    resolution_t resolution,
                                    const Context &ctx) const;

    // Stream simplify_path(const Partition& partition,
    //                      resolution_t resolution) const;
};

template <typename Impl>
Stream::Stream(Impl &&impl)
    : p_impl(new traits::remove_cv_t<Impl>(std::forward<Impl>(impl)))
{
}

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_STREAM_H_
