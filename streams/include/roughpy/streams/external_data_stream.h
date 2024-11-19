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

#ifndef ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
#define ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_

#include "stream_base.h"

#include "dyadic_caching_layer.h"
#include "stream.h"
#include <roughpy/core/traits.h>
#include <roughpy/platform.h>
#include <roughpy/platform/serialization.h>

#include <boost/url/url.hpp>

#include <memory>

namespace rpy {
namespace streams {

class ROUGHPY_STREAMS_EXPORT ExternalDataStreamSource
{

public:
    virtual ~ExternalDataStreamSource();

    virtual dimn_t
    query(scalars::KeyScalarArray& result, const intervals::Interval& interval,
          const StreamSchema& schema)
            = 0;


    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(ExternalDataStreamSource) {}


class ExternalDataStreamConstructor;

class ROUGHPY_STREAMS_EXPORT ExternalDataSourceFactory
{

public:
    using url = boost::urls::url;

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

class ROUGHPY_STREAMS_EXPORT ExternalDataStreamConstructor
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

class ROUGHPY_STREAMS_EXPORT ExternalDataStream : public DyadicCachingLayer
{
    std::unique_ptr<ExternalDataStreamSource> p_source;

public:
    using url = boost::urls::url;

    template <
            typename Source,
            typename
            = enable_if_t<is_base_of_v<ExternalDataStreamSource, Source>>>
    explicit ExternalDataStream(Source&& src, StreamMetadata md)
        : DyadicCachingLayer(std::move(md)),
          p_source(new Source(std::forward<Source>(src)))
    {}

    template <
            typename Source,
            typename
            = enable_if_t<is_base_of_v<ExternalDataStreamSource, Source>>>
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


    RPY_SERIAL_SERIALIZE_FN();
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

#ifdef RPY_COMPILING_STREAMS
RPY_SERIAL_EXTERN_SERIALIZE_CLS_BUILD(ExternalDataStream)
#else
RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMP(ExternalDataStream)
#endif

RPY_SERIAL_SERIALIZE_FN_IMPL(ExternalDataStream) {
    RPY_SERIAL_SERIALIZE_BASE(DyadicCachingLayer);

}

}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::streams::ExternalDataStream,
                            rpy::serial::specialization::member_serialize)

#endif// ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
