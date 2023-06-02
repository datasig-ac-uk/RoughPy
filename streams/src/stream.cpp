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




#include "stream.h"


using namespace rpy;
using namespace streams;

const algebra::Context& rpy::streams::Stream::get_default_context() const {
    RPY_CHECK(p_impl);
    return *p_impl->metadata().default_context;
}
const rpy::streams::StreamMetadata &rpy::streams::Stream::metadata() const {
    RPY_CHECK(p_impl);
    return p_impl->metadata();
}

const StreamSchema &Stream::schema() const {
    RPY_CHECK(p_impl);
    return p_impl->schema();
}

rpy::streams::Stream::Lie rpy::streams::Stream::log_signature() const {
    const auto& md = metadata();

    return p_impl->log_signature(md.effective_support,
                                 md.default_resolution,
                                 *md.default_context);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(const rpy::streams::Stream::Context &ctx) const {
    const auto& md = metadata();
    return p_impl->log_signature(md.effective_support,
                                 md.default_resolution,
                                 ctx);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(rpy::resolution_t resolution) {
    const auto& md = metadata();
    return p_impl->log_signature(md.effective_support,
                                 resolution,
                                 *md.default_context);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(rpy::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
    const auto& md = metadata();
    return p_impl->log_signature(md.effective_support,
                                 resolution,
                                 ctx);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(const rpy::streams::Stream::Interval &interval) const {
    const auto& md = metadata();
    return p_impl->log_signature(interval,
                                 md.default_resolution,
                                 *md.default_context);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(const rpy::streams::Stream::Interval &interval, rpy::resolution_t resolution) const {
    const auto& md = metadata();
    return p_impl->log_signature(interval,
                                 resolution,
                                 *md.default_context);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(const rpy::streams::Stream::Interval &interval, rpy::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
    return p_impl->log_signature(interval,
                                 resolution,
                                 ctx);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature() const {
    const auto& md = metadata();
    return p_impl->signature(md.effective_support,
                             md.default_resolution,
                             *md.default_context);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(const rpy::streams::Stream::Context &ctx) const {
    const auto& md = metadata();
    return p_impl->signature(md.effective_support,
                             md.default_resolution,
                             ctx);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(rpy::resolution_t resolution) {
    const auto& md = metadata();
    return p_impl->signature(md.effective_support,
                             resolution,
                             *md.default_context);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(rpy::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
    const auto& md = metadata();
    return p_impl->signature(md.effective_support,
                             resolution,
                             ctx);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(const rpy::streams::Stream::Interval &interval) const {
    const auto& md = metadata();
    return p_impl->signature(interval,
                             md.default_resolution,
                             *md.default_context);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(const rpy::streams::Stream::Interval &interval, rpy::resolution_t resolution) const {
    const auto& md = metadata();
    return p_impl->signature(interval,
                             resolution,
                             *md.default_context);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(const rpy::streams::Stream::Interval &interval, rpy::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
    return p_impl->signature(interval,
                             resolution,
                             ctx);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(const rpy::streams::Stream::Interval &domain, const rpy::streams::Stream::Lie &perturbation) const {
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info{
        log_signature(domain,
                      md.default_resolution,
                      *md.default_context),
        perturbation
    };

    return md.default_context->sig_derivative({std::move(info)}, md.cached_vector_type);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(const rpy::streams::Stream::Interval &domain, const rpy::streams::Stream::Lie &perturbation, const rpy::streams::Stream::Context &ctx) const {
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info {
        log_signature(domain,
                      md.default_resolution,
                      ctx),
        perturbation
    };

    return ctx.sig_derivative({std::move(info)}, md.cached_vector_type);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(const rpy::streams::Stream::Interval &domain, const rpy::streams::Stream::Lie &perturbation, rpy::resolution_t resolution) const {
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info {
        log_signature(domain,
                      resolution,
                      *md.default_context),
        perturbation
    };

    return md.default_context->sig_derivative({std::move(info)}, md.cached_vector_type);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(const rpy::streams::Stream::Interval &domain, const rpy::streams::Stream::Lie &perturbation, rpy::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info {
        log_signature(domain, resolution, ctx),
        perturbation
    };
    return ctx.sig_derivative({std::move(info)}, md.cached_vector_type);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(const rpy::streams::Stream::perturbation_list_t &perturbations, rpy::resolution_t resolution) const {
    const auto& md = metadata();
    return signature_derivative(perturbations, resolution, *md.default_context);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(const rpy::streams::Stream::perturbation_list_t &perturbations, rpy::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
    const auto &md = metadata();
    std::vector<algebra::DerivativeComputeInfo> info;
    info.reserve(perturbations.size());
    for (auto &&pert : perturbations) {
        info.push_back({
            log_signature(pert.first, resolution, ctx),
            pert.second});
    }
    return ctx.sig_derivative(info, md.cached_vector_type);
}


#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::Stream
#include <roughpy/platform/serialization_instantiations.inl>
