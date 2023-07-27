#ifndef ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
#define ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_

#include "stream_base.h"

#include "dyadic_caching_layer.h"
#include "stream.h"
#include <roughpy/core/traits.h>
#include <roughpy/platform.h>

#include <memory>

namespace rpy {
namespace streams {

class RPY_EXPORT ExternalDataStreamSource
{

public:
    virtual ~ExternalDataStreamSource();

    virtual dimn_t
    query(scalars::KeyScalarArray& result, const intervals::Interval& interval,
          const StreamSchema& schema)
            = 0;
};

class ExternalDataStreamConstructor;

class RPY_EXPORT ExternalDataSourceFactory
{

public:
    virtual void destroy_payload(void*& payload) const;

    virtual ~ExternalDataSourceFactory();

    virtual void set_width(void* payload, deg_t width) const;
    virtual void set_depth(void* payload, deg_t depth) const;
    virtual void
    set_ctype(void* payload, const scalars::ScalarType* ctype) const;
    virtual void set_context(void* payload, algebra::context_pointer ctx) const;
    virtual void
    set_support(void* payload, intervals::RealInterval support) const;
    virtual void set_vtype(void* payload, algebra::VectorType vtype) const;
    virtual void set_resolution(void* payload, resolution_t resolution) const;
    virtual void set_schema(void* payload, std::shared_ptr<StreamSchema>
            schema) const;

    virtual void
    add_option(void* payload, const string& option, void* value) const;

    virtual ExternalDataStreamConstructor get_constructor(const url& uri) const
            = 0;
    virtual Stream construct_stream(void* payload) const = 0;
};

class RPY_EXPORT ExternalDataStreamConstructor
{
    const ExternalDataSourceFactory* p_factory = nullptr;
    void* p_payload = nullptr;

public:
    ExternalDataStreamConstructor() = default;

    ExternalDataStreamConstructor(const ExternalDataStreamConstructor& other)
            = delete;
    ExternalDataStreamConstructor(ExternalDataStreamConstructor&& other
    ) noexcept;

    ExternalDataStreamConstructor(
            const ExternalDataSourceFactory* factory, void* payload
    );
    ~ExternalDataStreamConstructor();

    ExternalDataStreamConstructor&
    operator=(const ExternalDataStreamConstructor& other)
            = delete;
    ExternalDataStreamConstructor&
    operator=(ExternalDataStreamConstructor&& other) noexcept;

    void set_width(deg_t width);
    void set_depth(deg_t depth);
    void set_ctype(const scalars::ScalarType* ctype);
    void set_context(algebra::context_pointer ctx);
    void set_support(intervals::RealInterval support);
    void set_vtype(algebra::VectorType vtype);
    void set_resolution(resolution_t resolution);
    void set_schema(std::shared_ptr<StreamSchema> schema);

    void add_option(const string& option, void* value);

    RPY_NO_DISCARD Stream construct();

    operator bool() const noexcept { return p_factory != nullptr; }
};

class RPY_EXPORT ExternalDataStream : public DyadicCachingLayer
{
    std::unique_ptr<ExternalDataStreamSource> p_source;

public:
    template <
            typename Source,
            typename
            = enable_if_t<is_base_of<ExternalDataStreamSource, Source>::value>>
    explicit ExternalDataStream(Source&& src, StreamMetadata md)
        : DyadicCachingLayer(std::move(md)),
          p_source(new Source(std::forward<Source>(src)))
    {}

    template <
            typename Source,
            typename
            = enable_if_t<is_base_of<ExternalDataStreamSource, Source>::value>>
    explicit ExternalDataStream(Source&& src, StreamMetadata md,
                                std::shared_ptr<StreamSchema> schema)
        : DyadicCachingLayer(std::move(md), std::move(schema)),
          p_source(new Source(std::forward<Source>(src)))
    {}

    static void
    register_factory(std::unique_ptr<const ExternalDataSourceFactory>&& factory
    );
    static ExternalDataStreamConstructor get_factory_for(const url& uri);

protected:
    algebra::Lie log_signature_impl(
            const intervals::Interval& interval, const algebra::Context& ctx
    ) const override;
};

template <typename Source>
class RegisterExternalDataSourceFactoryHelper
{
public:
    template <typename... Args>
    RegisterExternalDataSourceFactoryHelper(Args&&... args)
    {
        ExternalDataStream::register_factory(
                std::unique_ptr<const ExternalDataSourceFactory>(
                        new Source(std::forward<Args>(args)...)
                )
        );
    }
};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
