


#include "stream.h"


using namespace rpy;
using namespace streams;

const algebra::Context& rpy::streams::Stream::get_default_context() const {
    return *p_impl->metadata().default_context;
}
const rpy::streams::StreamMetadata &rpy::streams::Stream::metadata() const noexcept {
    return p_impl->metadata();
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
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(rpy::streams::resolution_t resolution) {
    const auto& md = metadata();
    return p_impl->log_signature(md.effective_support,
                                 resolution,
                                 *md.default_context);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(rpy::streams::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
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
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(const rpy::streams::Stream::Interval &interval, rpy::streams::resolution_t resolution) const {
    const auto& md = metadata();
    return p_impl->log_signature(interval,
                                 resolution,
                                 *md.default_context);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(const rpy::streams::Stream::Interval &interval, rpy::streams::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
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
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(rpy::streams::resolution_t resolution) {
    const auto& md = metadata();
    return p_impl->signature(md.effective_support,
                             resolution,
                             *md.default_context);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(rpy::streams::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
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
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(const rpy::streams::Stream::Interval &interval, rpy::streams::resolution_t resolution) const {
    const auto& md = metadata();
    return p_impl->signature(interval,
                             resolution,
                             *md.default_context);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(const rpy::streams::Stream::Interval &interval, rpy::streams::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
    return p_impl->signature(interval,
                             resolution,
                             ctx);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(rpy::streams::Stream::Interval &domain, const rpy::streams::Stream::Lie &perturbation) const {
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info{
        log_signature(domain,
                      md.default_resolution,
                      *md.default_context),
        perturbation
    };

    return md.default_context->sig_derivative({std::move(info)}, md.cached_vector_type);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(rpy::streams::Stream::Interval &domain, const rpy::streams::Stream::Lie &perturbation, const rpy::streams::Stream::Context &ctx) const {
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info {
        log_signature(domain,
                      md.default_resolution,
                      ctx),
        perturbation
    };

    return ctx.sig_derivative({std::move(info)}, md.cached_vector_type);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(rpy::streams::Stream::Interval &domain, const rpy::streams::Stream::Lie &perturbation, rpy::streams::resolution_t resolution) const {
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info {
        log_signature(domain,
                      resolution,
                      *md.default_context),
        perturbation
    };

    return md.default_context->sig_derivative({std::move(info)}, md.cached_vector_type);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(rpy::streams::Stream::Interval &domain, const rpy::streams::Stream::Lie &perturbation, rpy::streams::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info {
        log_signature(domain, resolution, ctx),
        perturbation
    };
    return ctx.sig_derivative({std::move(info)}, md.cached_vector_type);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(const rpy::streams::Stream::perturbation_list_t &perturbations, rpy::streams::resolution_t resolution) const {
    const auto& md = metadata();
    return signature_derivative(perturbations, resolution, *md.default_context);
}
rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(const rpy::streams::Stream::perturbation_list_t &perturbations, rpy::streams::resolution_t resolution, const rpy::streams::Stream::Context &ctx) const {
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
