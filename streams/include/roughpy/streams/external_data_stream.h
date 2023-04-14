#ifndef ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
#define ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_

#include "stream_base.h"

#include <roughpy/core/traits.h>
#include <roughpy/platform.h>
#include "stream.h"

#include <memory>

namespace rpy { namespace streams {


class ROUGHPY_STREAMS_EXPORT ExternalDataStreamSource {

public:

    virtual ~ExternalDataStreamSource();

    virtual dimn_t query(scalars::KeyScalarArray& result, const intervals::Interval& interval) = 0;

};

class ROUGHPY_STREAMS_EXPORT ExternalDataSourceFactory {

public:
    virtual ~ExternalDataSourceFactory();


    virtual bool supports(const url& uri) const;
    virtual Stream construct_stream(const url& uri, StreamMetadata md) const = 0;

};

class ROUGHPY_STREAMS_EXPORT ExternalDataStream : public StreamInterface {
    std::unique_ptr<ExternalDataStreamSource> p_source;

public:
    template <typename Source, typename=traits::enable_if_t<traits::is_base_of<ExternalDataStreamSource, Source>::value>>
    explicit ExternalDataStream(Source &&src, StreamMetadata md)
        : StreamInterface(std::move(md)),
          p_source(new Source(std::forward<Source>(src)))
    {}

    static void register_factory(std::unique_ptr<const ExternalDataSourceFactory>&& factory);
    static const ExternalDataSourceFactory* get_factory_for(const url& uri);

protected:
    algebra::Lie log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const override;
};


template <typename Source>
class RegisterExternalDataSourceFactoryHelper {
public:

    template <typename... Args>
    RegisterExternalDataSourceFactoryHelper(Args&&... args) {
        ExternalDataStream::register_factory(
            std::unique_ptr<const ExternalDataSourceFactory>(new Source(std::forward<Args>(args)...)));
    }
};


}}

#endif // ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
