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

    FreeTensor signature_derivative(Interval &domain,
                                    const Lie &perturbation) const;
    FreeTensor signature_derivative(Interval &domain,
                                    const Lie &perturbation,
                                    const Context &ctx) const;
    FreeTensor signature_derivative(Interval &domain,
                                    const Lie &perturbation,
                                    resolution_t resolution) const;
    FreeTensor signature_derivative(Interval &domain,
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
    : p_impl(new traits::remove_cv_ref_t<Impl>(std::forward<Impl>(impl)))
{
}

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_STREAM_H_
