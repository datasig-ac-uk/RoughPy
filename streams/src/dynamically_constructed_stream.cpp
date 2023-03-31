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
// Created by user on 18/03/23.
//

#include "dynamically_constructed_stream.h"

using namespace rpy;
using namespace rpy::streams;

algebra::Lie DynamicallyConstructedStream::log_signature(const intervals::DyadicInterval &interval, resolution_t resolution, const algebra::Context &ctx) const {
    const auto& md = metadata();
    if (!interval.intersects_with(md.effective_support)) {
        return ctx.zero_lie(md.cached_vector_type);
    }
    // TODO: Logic for truncating
    return DyadicCachingLayer::log_signature(interval, resolution, ctx);
}
algebra::Lie DynamicallyConstructedStream::log_signature(const intervals::Interval &domain, resolution_t resolution, const algebra::Context &ctx) const {
    const auto& md = metadata();
    if (!domain.intersects_with(md.effective_support)) {
        return ctx.zero_lie(md.cached_vector_type);
    }
    // TODO: logic for truncating
    return DyadicCachingLayer::log_signature(domain, resolution, ctx);
}
