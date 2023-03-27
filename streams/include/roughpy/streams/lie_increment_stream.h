#ifndef ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_
#define ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_

#include "stream_base.h"
#include "dyadic_caching_layer.h"

#include <boost/container/flat_map.hpp>

#include <roughpy/core/helpers.h>
#include <roughpy/scalars/key_scalar_array.h>

namespace rpy { namespace streams {

class ROUGHPY_STREAMS_EXPORT LieIncrementStream : public DyadicCachingLayer {
    scalars::KeyScalarArray m_buffer;
    boost::container::flat_map<param_t, dimn_t> m_mapping;

    using base_t = DyadicCachingLayer;
public:

    LieIncrementStream(
        scalars::KeyScalarArray&& buffer,
        Slice<param_t> indices,
        StreamMetadata md
        );

    bool empty(const intervals::Interval &interval) const noexcept override;

protected:
    algebra::Lie log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const override;

};


}}


#endif // ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_
