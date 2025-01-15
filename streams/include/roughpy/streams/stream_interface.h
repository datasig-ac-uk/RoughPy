#ifndef STREAM_INTERFACE_H
#define STREAM_INTERFACE_H


#include "roughpy/core/types.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/check.h"
#include "roughpy/core/smart_ptr.h"

#include "roughpy/platform/alloc.h"
#include "roughpy/platform/serialization.h"

#include "roughpy/intervals/dyadic_interval.h"
#include "roughpy/intervals/real_interval.h"
#include "roughpy/intervals/interval.h"
#include "roughpy/intervals/partition.h"

#include "roughpy/algebra/lie.h"
#include "roughpy/algebra/free_tensor.h"

#include "stream_metadata.h"
#include "roughpy_streams_export.h"

namespace rpy {
namespace streams {


RPY_NO_DISCARD
inline
algebra::FreeTensor lie_to_tensor(const algebra::Lie& lie)
{
    auto ctx = lie.context();
    return ctx->lie_to_tensor(lie);
}

RPY_NO_DISCARD
inline
algebra::Lie tensor_to_lie(const algebra::FreeTensor& tensor)
{
    auto ctx = tensor.context();
    return ctx->tensor_to_lie(tensor);
}


class ROUGHPY_STREAMS_EXPORT StreamInterface : public mem::SmallObjectBase
{
public:
    using Lie = algebra::Lie;
    using FreeTensor = algebra::FreeTensor;
    using Context = algebra::Context;
    using Interval = intervals::Interval;
    using RealInterval = intervals::RealInterval;
    using DyadicInterval = intervals::DyadicInterval;

    virtual ~StreamInterface();

    // Access to stream information
    RPY_NO_DISCARD
    virtual std::shared_ptr<const StreamMetadata>
    metadata() const noexcept = 0;

    RPY_NO_DISCARD
    virtual const intervals::RealInterval& support() const noexcept;

    // Signature and log-signature computations
    RPY_NO_DISCARD
    virtual Lie log_signature(const DyadicInterval& interval,
                              resolution_t resolution,
                              const Context& context) const = 0;

    RPY_NO_DISCARD
    virtual Lie log_signature(const Interval& interval,
                              resolution_t resolution,
                              const Context& context) const = 0;

    RPY_NO_DISCARD Lie log_signature(const Interval& interval,
                                     resolution_t resolution) const
    {
        return log_signature(interval,
                             resolution,
                             *metadata()->default_context());
    }

    RPY_NO_DISCARD Lie log_signature(const Interval& interval,
                                     const Context& context) const
    {
        return log_signature(interval, metadata()->resolution(), context);
    }

    RPY_NO_DISCARD Lie log_signature(const Interval& interval) const
    {
        const auto md = *metadata();
        return log_signature(interval, md.resolution(), *md.default_context());
    }

    RPY_NO_DISCARD Lie log_signature() const
    {
        const auto& md = *metadata();
        return log_signature(support(), md.resolution(), *md.default_context());
    }

    RPY_NO_DISCARD
    FreeTensor signature(const Interval& interval,
                         resolution_t resolution,
                         const Context& context) const;

    RPY_NO_DISCARD
    FreeTensor signature(const Interval& interval,
                         resolution_t resolution) const
    {
        return signature(interval, resolution, *metadata()->default_context());
    }

    RPY_NO_DISCARD
    FreeTensor signature(const Interval& interval,
                         const Context& context) const
    {
        return signature(interval, metadata()->resolution(), context);
    }

    RPY_NO_DISCARD
    FreeTensor signature(const Interval& interval) const
    {
        const auto md = *metadata();
        return signature(interval,
                         md.resolution(),
                         *md.default_context());
    }

    RPY_NO_DISCARD
    FreeTensor signature() const
    {
        const auto& md = *metadata();
        return signature(support(), md.resolution(), *md.default_context());
    }


    RPY_NO_DISCARD
    FreeTensor signature_derivative(const Interval& interval,
                                    const Lie& perturbation,
                                    resolution_t resolution,
                                    const Context& ctx) const;

    RPY_NO_DISCARD
    FreeTensor signature_derivative(const Interval& interval,
                                    const Lie& perturbation) const
    {
        const auto md = *metadata();
        return signature_derivative(interval,
                                    perturbation,
                                    md.resolution(),
                                    *md.default_context());
    }

    RPY_NO_DISCARD
    FreeTensor signature_derivative(const Interval& interval,
                                    const Lie& perturbation,
                                    resolution_t resolution) const
    {
        return signature_derivative(interval,
                                    perturbation,
                                    resolution,
                                    *metadata()->default_context());
    }

    RPY_NO_DISCARD
    FreeTensor signature_derivative(const Interval& interval,
                                    const Lie& perturbation,
                                    const Context& ctx) const
    {
        return signature_derivative(interval,
                                    perturbation,
                                    metadata()->resolution(),
                                    ctx);
    }

    RPY_NO_DISCARD
    FreeTensor signature_derivative(
        span<pair<RealInterval, Lie> > perturbations,
        resolution_t resolution,
        const Context& ctx) const;

    RPY_NO_DISCARD
    FreeTensor signature_derivative(
        span<pair<RealInterval, Lie> > perturbations)
    {
        const auto& md = *metadata();
        return signature_derivative(perturbations,
                                    md.resolution(),
                                    *md.default_context());
    }

    RPY_NO_DISCARD
    FreeTensor signature_derivative(
        span<pair<RealInterval, Lie> > perturbations,
        resolution_t resolution) const
    {
        return signature_derivative(perturbations,
                                    resolution,
                                    *metadata()->default_context());
    }

    RPY_NO_DISCARD
    FreeTensor signature_derivative(
        span<pair<RealInterval, Lie> > perturbations,
        const Context& ctx) const
    {
        return signature_derivative(perturbations,
                                    metadata()->resolution(),
                                    ctx);
    }


    // stream modifications

    RPY_NO_DISCARD
    virtual
    std::shared_ptr<const StreamInterface> clone() const = 0;

protected:
    // helpers
    RPY_NO_DISCARD FreeTensor unit_tensor() const;

    RPY_NO_DISCARD Lie zero_lie() const;

public:
    RPY_SERIAL_SERIALIZE_FN() {}
};


RPY_NO_DISCARD ROUGHPY_STREAMS_EXPORT
std::shared_ptr<const StreamInterface> restrict(
    std::shared_ptr<const StreamInterface> stream,
    const intervals::Interval& interval);

RPY_NO_DISCARD ROUGHPY_STREAMS_EXPORT
std::shared_ptr<const StreamInterface> simplify(
    std::shared_ptr<const StreamInterface> stream,
    const intervals::Partition& partition,
    resolution_t resolution,
    const algebra::Context& ctx);


}// streams
}// rpy

#endif //STREAM_INTERFACE_H