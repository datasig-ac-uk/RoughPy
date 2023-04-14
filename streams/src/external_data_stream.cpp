// Copyright (c) 2023 Datasig Group. All rights reserved.
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

//
// Created by user on 13/04/23.
//


#include "external_data_stream.h"

#include <vector>
#include <memory>
#include <mutex>

using namespace rpy;


streams::ExternalDataStreamSource::~ExternalDataStreamSource() = default;

streams::ExternalDataSourceFactory::~ExternalDataSourceFactory() = default;

bool streams::ExternalDataSourceFactory::supports(const url &uri) const {
    return false;
}

algebra::Lie streams::ExternalDataStream::log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const {
    scalars::KeyScalarArray buffer(ctx.ctype());
    auto num_increments = p_source->query(buffer, interval);

    algebra::SignatureData tmp {
        scalars::ScalarStream(ctx.ctype()),
        std::vector<const key_type*>(),
        metadata().cached_vector_type
    };

    tmp.data_stream.reserve_size(num_increments);
    const auto width = static_cast<dimn_t>(metadata().width);

    scalars::ScalarPointer buf_ptr(buffer);
    for (dimn_t i=0; i<num_increments; ++i) {
        tmp.data_stream.push_back({buf_ptr, width});
        buf_ptr += width;
    }

    return ctx.log_signature(tmp);
}


static std::mutex s_factory_guard;
static std::vector<std::unique_ptr<const streams::ExternalDataSourceFactory>> s_factory_list;


void streams::ExternalDataStream::register_factory(std::unique_ptr<const ExternalDataSourceFactory> &&factory) {
    std::lock_guard<std::mutex> access(s_factory_guard);
    s_factory_list.push_back(std::move(factory));
}
const streams::ExternalDataSourceFactory * streams::ExternalDataStream::get_factory_for(const url& uri) {
    std::lock_guard<std::mutex> access(s_factory_guard);

    for (auto it=s_factory_list.rbegin(); it != s_factory_list.rend(); ++it) {
        if ((*it)->supports(uri)) {
            return it->get();
        }
    }

    return nullptr;
}
