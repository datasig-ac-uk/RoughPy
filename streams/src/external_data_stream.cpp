// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 13/04/23.
//

#include <roughpy/streams/external_data_stream.h>

#include "external_data_sources/csv_data_source.h"
#include "external_data_sources/sound_file_data_source.h"

#include <memory>
#include <mutex>
#include <vector>

using namespace rpy;

streams::ExternalDataStreamSource::~ExternalDataStreamSource() = default;

streams::ExternalDataSourceFactory::~ExternalDataSourceFactory() = default;

algebra::Lie streams::ExternalDataStream::log_signature_impl(
        const intervals::Interval& interval, const algebra::Context& ctx) const
{
    scalars::KeyScalarArray buffer(ctx.ctype());
    auto num_increments = p_source->query(buffer, interval, schema());

    algebra::SignatureData tmp{scalars::ScalarStream(ctx.ctype()),
                               std::vector<const key_type*>(),
                               metadata().cached_vector_type};

    tmp.data_stream.reserve_size(num_increments);
    const auto width = static_cast<dimn_t>(metadata().width);

    scalars::ScalarPointer buf_ptr(buffer);
    for (dimn_t i = 0; i < num_increments; ++i) {
        tmp.data_stream.push_back({buf_ptr, width});
        buf_ptr += width;
    }

    return ctx.log_signature(tmp);
}

static std::mutex s_factory_guard;
static std::vector<std::unique_ptr<const streams::ExternalDataSourceFactory>>
        s_factory_list;

void streams::ExternalDataStream::register_factory(
        std::unique_ptr<const ExternalDataSourceFactory>&& factory)
{
    std::lock_guard<std::mutex> access(s_factory_guard);
    s_factory_list.push_back(std::move(factory));
}
streams::ExternalDataStreamConstructor
streams::ExternalDataStream::get_factory_for(const url& uri)
{
    std::lock_guard<std::mutex> access(s_factory_guard);

    if (s_factory_list.empty()) {
        // Register defaults
        s_factory_list.emplace_back(new SoundFileDataSourceFactory);
    }

    ExternalDataStreamConstructor ctor;
    for (auto it = s_factory_list.rbegin(); it != s_factory_list.rend(); ++it) {
        ctor = (*it)->get_constructor(uri);
        if (ctor) { return ctor; }
    }

    return ctor;
}

void streams::ExternalDataSourceFactory::destroy_payload(void*& payload) const
{
    RPY_DBG_ASSERT(payload == nullptr);
}

void streams::ExternalDataSourceFactory::set_width(void* payload,
                                                   deg_t width) const
{}
void streams::ExternalDataSourceFactory::set_depth(void* payload,
                                                   deg_t depth) const
{}
void streams::ExternalDataSourceFactory::set_ctype(
        void* payload, const scalars::ScalarType* ctype) const
{}
void streams::ExternalDataSourceFactory::set_context(
        void* payload, algebra::context_pointer ctx) const
{}
void streams::ExternalDataSourceFactory::set_support(
        void* payload, intervals::RealInterval support) const
{}
void streams::ExternalDataSourceFactory::set_vtype(
        void* payload, algebra::VectorType vtype) const
{}
void streams::ExternalDataSourceFactory::set_resolution(
        void* payload, resolution_t resolution) const
{}
void streams::ExternalDataSourceFactory::set_schema(
        void* payload, std::shared_ptr<StreamSchema> schema
) const
{}

void streams::ExternalDataSourceFactory::add_option(void* payload,
                                                    const string& option,
                                                    void* value) const
{}

streams::ExternalDataStreamConstructor::ExternalDataStreamConstructor(
        const streams::ExternalDataSourceFactory* factory, void* payload)
    : p_factory(factory), p_payload(payload)
{
    RPY_CHECK(p_factory != nullptr && p_payload != nullptr);
}

streams::ExternalDataStreamConstructor::ExternalDataStreamConstructor(
        streams::ExternalDataStreamConstructor&& other) noexcept
    : p_factory(other.p_factory), p_payload(other.p_payload)
{
    other.p_factory = nullptr;
    other.p_payload = nullptr;
}

streams::ExternalDataStreamConstructor::~ExternalDataStreamConstructor()
{
    if (p_factory != nullptr && p_payload != nullptr) {
        p_factory->destroy_payload(p_payload);
    }
}
streams::ExternalDataStreamConstructor&
streams::ExternalDataStreamConstructor::operator=(
        streams::ExternalDataStreamConstructor&& other) noexcept
{
    if (this != &other) {
        this->~ExternalDataStreamConstructor();
        p_factory = other.p_factory;
        p_payload = other.p_payload;
        other.p_payload = nullptr;
        other.p_factory = nullptr;
    }
    return *this;
}

void streams::ExternalDataStreamConstructor::set_width(deg_t width)
{
    p_factory->set_width(p_payload, width);
}
void streams::ExternalDataStreamConstructor::set_depth(deg_t depth)
{
    p_factory->set_depth(p_payload, depth);
}
void streams::ExternalDataStreamConstructor::set_ctype(
        const scalars::ScalarType* ctype)
{
    p_factory->set_ctype(p_payload, ctype);
}
void streams::ExternalDataStreamConstructor::set_context(
        algebra::context_pointer ctx)
{
    p_factory->set_context(p_payload, std::move(ctx));
}
void streams::ExternalDataStreamConstructor::set_support(
        intervals::RealInterval support)
{
    p_factory->set_support(p_payload, std::move(support));
}
void streams::ExternalDataStreamConstructor::set_vtype(
        algebra::VectorType vtype)
{
    p_factory->set_vtype(p_payload, vtype);
}
void streams::ExternalDataStreamConstructor::set_resolution(
        resolution_t resolution)
{
    p_factory->set_resolution(p_payload, resolution);
}
void streams::ExternalDataStreamConstructor::set_schema(
        std::shared_ptr<StreamSchema> schema
)
{
    p_factory->set_schema(p_payload, std::move(schema));
}

void streams::ExternalDataStreamConstructor::add_option(const string& option,
                                                        void* value)
{
    p_factory->add_option(p_payload, option, value);
}
streams::Stream streams::ExternalDataStreamConstructor::construct()
{
    auto* payload = p_payload;
    p_payload = nullptr;
    return p_factory->construct_stream(payload);
}
