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
    virtual const std::shared_ptr<StreamMetadata>&
    metadata() const noexcept = 0;

    RPY_NO_DISCARD
    const intervals::RealInterval&
    domain() const noexcept
    {
        return metadata()->domain();
    }

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
                                     resolution_t resolution) const;

    RPY_NO_DISCARD Lie log_signature(const Interval& interval,
                                     const Context& context) const;

    RPY_NO_DISCARD Lie log_signature(const Interval& interval) const;

    RPY_NO_DISCARD
    FreeTensor signature(const Interval& interval,
                         resolution_t resolution,
                         const Context& context) const;

    RPY_NO_DISCARD
    FreeTensor signature(const Interval& interval,
                         resolution_t resolution) const;

    RPY_NO_DISCARD
    FreeTensor signature(const Interval& interval,
                         const Context& context) const;

    RPY_NO_DISCARD
    FreeTensor signature(const Interval& interval) const;

protected:
    // helpers
    RPY_NO_DISCARD FreeTensor unit_tensor() const;
    RPY_NO_DISCARD Lie zero_lie() const;

    RPY_SERIAL_SERIALIZE_FN() {}
};


}// streams
}// rpy

#endif //STREAM_INTERFACE_H