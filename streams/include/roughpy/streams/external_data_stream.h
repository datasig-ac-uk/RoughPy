#ifndef ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
#define ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_

#include "stream_base.h"

#include <roughpy/core/traits.h>

#include <memory>

namespace rpy { namespace streams {


class ROUGHPY_STREAMS_EXPORT ExternalDataStreamSource {

public:

    virtual ~ExternalDataStreamSource();

    virtual scalars::KeyScalarArray query(const intervals::Interval& interval, const scalars::ScalarType* ctype) = 0;

};


class ROUGHPY_STREAMS_EXPORT ExternalDataStream : public StreamInterface {
    std::unique_ptr<ExternalDataStreamSource> p_source;

public:
    template <typename Source, typename=traits::enable_if_t<traits::is_base_of<ExternalDataStreamSource, Source>::value>>
    explicit ExternalDataStream(Source &&src, StreamMetadata md)
        : StreamInterface(std::move(md)),
          p_source(new Source(std::move(src)))
    {}

protected:
    algebra::Lie log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const override;
};


}}

#endif // ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
