#include "stream_interface.h"

#include "piecewise_abelian_stream.h"
#include "restriction_stream.h"


using namespace rpy;
using namespace rpy::streams;

StreamInterface::~StreamInterface() = default;


StreamInterface::FreeTensor StreamInterface::signature(
    const Interval& interval,
    resolution_t resolution,
    const Context& context) const
{
    return lie_to_tensor(log_signature(interval, resolution, context)).exp();
}


StreamInterface::FreeTensor StreamInterface::unit_tensor() const
{
    const auto& md = metadata();
    algebra::VectorConstructionData data{
            scalars::KeyScalarArray(md->scalar_type()),
            algebra::VectorType::Dense
    };

    data.data.allocate_scalars(1);
    data.data[0] = scalars::Scalar(md->scalar_type(), 1, 1);

    return metadata()->default_context()->construct_free_tensor(data);
}


const intervals::RealInterval& StreamInterface::support() const noexcept
{
    return metadata()->domain();
}

StreamInterface::FreeTensor StreamInterface::signature_derivative(
    const Interval& interval,
    const Lie& perturbation,
    resolution_t resolution,
    const Context& ctx) const {}

StreamInterface::Lie StreamInterface::zero_lie() const
{
    const auto ctx = metadata()->default_context();
    return ctx->zero_lie(algebra::VectorType::Dense);
}

StreamInterface::FreeTensor StreamInterface::signature_derivative(
    span<pair<RealInterval, Lie> > perturbations,
    resolution_t resolution,
    const Context& ctx) const {}


std::shared_ptr<const StreamInterface> streams::restrict(
    std::shared_ptr<const StreamInterface> stream,
    const intervals::Interval& interval)
{
    return std::make_shared<RestrictionStream>(std::move(stream),
                                               intervals::RealInterval(
                                                   interval));
}

std::shared_ptr<const StreamInterface> streams::simplify(
    std::shared_ptr<const StreamInterface> stream,
    const intervals::Partition& partition,
    resolution_t resolution,
    const algebra::Context& ctx)
{
    using LiePiece = typename PiecewiseAbelianStream::LiePiece;
    std::vector<LiePiece> pieces;
    for (const auto& interval : partition) {
        pieces.emplace_back(interval, stream->log_signature(interval, resolution, ctx));
    }

    auto md_builder = StreamMetadata::builder(stream->metadata().get());

    md_builder.set_context(&ctx);
    md_builder.set_resolution(resolution);
    md_builder.set_domain(intervals::RealInterval(partition));

    return std::make_shared<PiecewiseAbelianStream>(std::move(pieces), md_builder.build());
}