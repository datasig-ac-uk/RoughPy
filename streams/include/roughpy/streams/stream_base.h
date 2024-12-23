// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_STREAMS_STREAM_BASE_H_
#define ROUGHPY_STREAMS_STREAM_BASE_H_

#include <roughpy/core/debug_assertion.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "roughpy/intervals/dyadic_interval.h"
#include <roughpy/algebra/context.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/shuffle_tensor.h>
#include <roughpy/intervals/real_interval.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/platform/errors.h>
#include "roughpy/platform/alloc.h"

#include "schema.h"

#include "roughpy_streams_export.h"

namespace rpy {
namespace streams {

/**
 * @brief Metadata associated with all path objects
 *
 * This struct holds various pieces of data about the space in which a path
 * lies: the underlying vector space dimension; the type of coefficients; the
 * effective support of the path (where the values of the path are
 * concentrated); the truncation depth for signatures and log-signatures; how
 * the data is stored in the path; and the storage model for the free tensor
 * signatures and Lie log- signatures.
 */
struct StreamMetadata {
    deg_t width;
    intervals::RealInterval effective_support;
    algebra::context_pointer default_context;
    const scalars::ScalarType* data_scalar_type;
    algebra::VectorType cached_vector_type;
    resolution_t default_resolution;
    intervals::IntervalType interval_type;
};


RPY_NO_DISCARD
inline resolution_t param_to_resolution(param_t arg) noexcept
{
    int exponent;
    frexp(arg, &exponent);
    /*
     * frexp returns fractional part in the range [0.5, 1), so the correct power
     * of 2 is actually exponent - 1.\
     */
    return -std::min(0, exponent - 1);
}



/**
 * @brief Base class for all stream types.
 *
 * An abstract stream provides methods for querying the signature or
 * log-signature over an interval in the parameter space, returning either
 * a free tensor or Lie element. This base class has establishes this interface
 * and also acts as a holder for the stream metadata.
 *
 * Stream implementations should implement the `log_signature_impl` virtual
 * function (taking `interval` and `context` arguments) that is used to
 * implement the other flavours of computation methods. (Note that signatures a
 * computed from log signatures, rather than using the data to compute these
 * independently.)
 */
class ROUGHPY_STREAMS_EXPORT StreamInterface : public mem::SmallObjectBase
{
    StreamMetadata m_metadata;
    std::shared_ptr<StreamSchema> p_schema;

public:
    RPY_NO_DISCARD const StreamMetadata& metadata() const noexcept
    {
        return m_metadata;
    }
    RPY_NO_DISCARD const StreamSchema& schema() const noexcept
    {
        return *p_schema;
    }

    explicit StreamInterface(
            StreamMetadata md,
            std::shared_ptr<StreamSchema> schema
    )
        : m_metadata(std::move(md)),
          p_schema(std::move(schema))
    {
        RPY_DBG_ASSERT(p_schema);
    }

    explicit StreamInterface(StreamMetadata md)
        : m_metadata(std::move(md)),
          p_schema(new StreamSchema(m_metadata.width))
    {}

    virtual ~StreamInterface() noexcept;
    RPY_NO_DISCARD virtual bool empty(const intervals::Interval& interval
    ) const noexcept;
    std::shared_ptr<StreamSchema> get_schema() const noexcept
    {
        return p_schema;
    }

protected:
    StreamInterface() : m_metadata(), p_schema() {}

    void set_metadata(StreamMetadata&& md) noexcept;

    void set_schema(std::shared_ptr<StreamSchema> schema) noexcept
    {
        p_schema = std::move(schema);
    }

    RPY_NO_DISCARD virtual algebra::Lie log_signature_impl(
            const intervals::Interval& interval,
            const algebra::Context& ctx
    ) const = 0;

    algebra::FreeTensor unit_tensor() const;

public:
    RPY_NO_DISCARD virtual algebra::Lie log_signature(
            const intervals::Interval& interval,
            const algebra::Context& ctx
    ) const;

    RPY_NO_DISCARD virtual algebra::Lie log_signature(
            const intervals::DyadicInterval& interval,
            resolution_t resolution,
            const algebra::Context& ctx
    ) const;

    RPY_NO_DISCARD virtual algebra::Lie log_signature(
            const intervals::Interval& interval,
            resolution_t resolution,
            const algebra::Context& ctx
    ) const;

    RPY_NO_DISCARD virtual algebra::FreeTensor signature(
            const intervals::Interval& interval,
            const algebra::Context& ctx
    ) const;

    RPY_NO_DISCARD virtual algebra::FreeTensor signature(
            const intervals::Interval& interval,
            resolution_t resolution,
            const algebra::Context& ctx
    ) const;

protected:
    // TODO: add methods for batch computing signatures via a computation tree

public:
    RPY_SERIAL_SERIALIZE_FN();
    RPY_SERIAL_ACCESS();
};

#ifdef RPY_COMPILING_STREAMS
RPY_SERIAL_EXTERN_SERIALIZE_CLS_BUILD(StreamInterface)
#else
RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMP(StreamInterface)
#endif

RPY_SERIAL_SERIALIZE_FN_IMPL(StreamInterface)
{
    RPY_SERIAL_SERIALIZE_NVP("metadata", m_metadata);
    RPY_SERIAL_SERIALIZE_NVP("schema", p_schema);
}

}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::streams::StreamInterface,
        rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_STREAMS_STREAM_BASE_H_
