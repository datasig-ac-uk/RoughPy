#ifndef ROUGHPY_STREAMS_STREAM_BASE_H_
#define ROUGHPY_STREAMS_STREAM_BASE_H_

#include <roughpy/algebra/context.h>
#include <roughpy/config/implementation_types.h>
#include <roughpy/intervals/real_interval.h>

#include "roughpy/intervals/dyadic_interval.h"
#include "roughpy_streams_export.h"

namespace rpy {
namespace streams {

using resolution_t = int;
using accuracy_t = param_t;

/**
 * @brief Metadata associated with all path objects
 *
 * This struct holds various pieces of data about the space in which a path
 * lies: the underlying vector space dimension; the type of coefficients; the
 * effective support of the path (where the values of the path are concentrated);
 * the truncation depth for signatures and log-signatures; how the data is stored
 * in the path; and the storage model for the free tensor signatures and Lie log-
 * signatures.
 */
struct StreamMetadata {
    deg_t width;
    intervals::RealInterval effective_support;
    algebra::context_pointer default_context;
    const scalars::ScalarType *data_scalar_type;
    algebra::VectorType cached_vector_type;
    resolution_t default_resolution;
};

/**
 * @brief Base class for all stream types.
 *
 * An abstract stream provides methods for querying the signature or
 * log-signature over an interval in the parameter space, returning either
 * a free tensor or Lie element. This base class has establishes this interface
 * and also acts as a holder for the stream metadata.
 *
 * Stream implementations should implement the `log_signature_impl` virtual function
 * (taking `interval` and `context` arguments) that is used to implement the
 * other flavours of computation methods. (Note that signatures a computed
 * from log signatures, rather than using the data to compute these
 * independently.)
 */
class ROUGHPY_STREAMS_EXPORT StreamInterface {
    StreamMetadata m_metadata;
public:

    const StreamMetadata& metadata() const noexcept;

    explicit StreamInterface(StreamMetadata md) : m_metadata(std::move(md))
    {}

    virtual bool empty(const intervals::Interval& interval) const noexcept;

protected:
    virtual algebra::Lie
    log_signature_impl(const intervals::Interval& interval,
                       const algebra::Context& ctx) const = 0;

public:

    virtual algebra::Lie
    log_signature(const intervals::Interval& interval,
                  const algebra::Context& ctx) const;

    virtual algebra::Lie
    log_signature(const intervals::DyadicInterval& interval,
                  resolution_t resolution,
                  const algebra::Context& ctx) const;

    virtual algebra::Lie
    log_signature(const intervals::Interval& interval,
                  resolution_t resolution,
                  const algebra::Context& ctx) const;

    virtual algebra::FreeTensor
    signature(const intervals::Interval& interval,
              const algebra::Context& ctx) const;

    virtual algebra::FreeTensor
    signature(const intervals::Interval& interval,
              resolution_t resolution,
              const algebra::Context& ctx) const;

protected:

    // TODO: add methods for batch computing signatures via a computation tree



};

/**
 * @brief Subclass of `StreamInterface` for solutions of controlled differential equations.
 */
class ROUGHPY_STREAMS_EXPORT SolutionStreamInterface : public StreamInterface {
public:
    using StreamInterface::StreamInterface;
    virtual algebra::Lie base_point() const = 0;
};






}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_STREAM_BASE_H_
