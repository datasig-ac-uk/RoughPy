#include "stream_interface.h"



using namespace rpy;
using namespace rpy::streams;

StreamInterface::~StreamInterface() = default;

StreamInterface::Lie StreamInterface::log_signature(const Interval& interval,
    resolution_t resolution) const
{
    return log_signature(interval, resolution, *metadata()->default_context());
}

StreamInterface::Lie StreamInterface::log_signature(const Interval& interval, const Context& context) const
{
    return log_signature(interval, metadata()->resolution(), context);
}

StreamInterface::Lie StreamInterface::log_signature(
    const Interval& interval) const
{
    const auto& md = metadata();
    return log_signature(interval, md->resolution(), *md->default_context());
}

StreamInterface::FreeTensor StreamInterface::signature(const Interval& interval, resolution_t resolution, const Context& context) const
{
    return lie_to_tensor(log_signature(interval, resolution, context)).exp();
}

StreamInterface::FreeTensor StreamInterface::signature(const Interval& interval, const Context& context) const
{
    return lie_to_tensor(log_signature(interval, context)).exp();
}

StreamInterface::FreeTensor StreamInterface::signature(const Interval& interval,
    resolution_t resolution) const
{
    return lie_to_tensor(log_signature(interval, resolution)).exp();
}

StreamInterface::FreeTensor StreamInterface::signature(
    const Interval& interval) const
{
    return lie_to_tensor(log_signature(interval)).exp();
}


StreamInterface::FreeTensor StreamInterface::unit_tensor() const
{
    const auto& md = metadata();
    algebra::VectorConstructionData data {
        scalars::KeyScalarArray (md->scalar_type()),
        algebra::VectorType::Dense
    };

    data.data.allocate_scalars(1);
    data.data[0] = scalars::Scalar(md->scalar_type(), 1, 1);

    return metadata()->default_context()->construct_free_tensor(data);
}
