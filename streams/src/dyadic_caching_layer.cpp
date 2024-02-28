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

//
// Created by user on 10/03/23.
//

#include <roughpy/streams/dyadic_caching_layer.h>

#include <roughpy/platform/archives.h>

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <cereal/types/map.hpp>

// This is not a good solution, but for now I don't see another way to force
// a link to bcrypt.
#ifdef RPY_PLATFORM_WINDOWS
#pragma comment(lib, "bcrypt.lib")
#endif

#include <fstream>

using namespace rpy;
using namespace rpy::streams;

DyadicCachingLayer::DyadicCachingLayer()
    : StreamInterface(), m_cache(), m_compute_lock(),
          m_cache_id(uuids::random_generator()())
{
//    auto path = get_config().stream_cache_dir() / to_string(m_cache_id);
//    {
//        std::ofstream fs(path);
//    }
//
//    m_file_lock = boost::interprocess::file_lock(path.c_str());
}

DyadicCachingLayer::DyadicCachingLayer(DyadicCachingLayer&& other) noexcept
    : StreamInterface(static_cast<StreamInterface&&>(other))
{
    std::lock_guard<std::recursive_mutex> access(other.m_compute_lock);
    m_cache = std::move(other.m_cache);
}
DyadicCachingLayer&
DyadicCachingLayer::operator=(DyadicCachingLayer&& other) noexcept
{
    if (&other != this) {
        std::lock_guard<std::recursive_mutex> this_access(m_compute_lock);
        std::lock_guard<std::recursive_mutex> that_access(other.m_compute_lock);
        m_cache = std::move(other.m_cache);
        StreamInterface::operator=(std::move(other));
    }
    return *this;
}
algebra::Lie
DyadicCachingLayer::log_signature(const intervals::DyadicInterval& interval,
                                  resolution_t resolution,
                                  const algebra::Context& ctx) const
{
    if (empty(interval)) {
        return ctx.zero_lie(DyadicCachingLayer::metadata().cached_vector_type);
    }

    const auto stream_resolution = metadata().default_resolution;

    if (interval.power() == stream_resolution) {
        std::lock_guard<std::recursive_mutex> access(m_compute_lock);

        auto& cached = m_cache[interval];
        if (!cached) { cached = log_signature_impl(interval, ctx); }
        // Currently, const borrowing is not permitted, so return a mutable
        // view.
        return cached.borrow_mut();
    }

    if (interval.power() > stream_resolution) {
        intervals::DyadicInterval tmp(interval);
        tmp.expand_interval(interval.power() - stream_resolution);
        RPY_DBG_ASSERT(tmp.power() == stream_resolution);
        if (rational_equals(tmp, interval)) {
            return log_signature(tmp, stream_resolution, ctx);
        }
        return algebra::Lie();
    }

    intervals::DyadicInterval lhs_itvl(interval);
    intervals::DyadicInterval rhs_itvl(interval);
    lhs_itvl.shrink_interval_left();
    rhs_itvl.shrink_interval_right();

    auto lhs = log_signature(lhs_itvl, resolution, ctx);
    auto rhs = log_signature(rhs_itvl, resolution, ctx);

    if (lhs.is_zero()) { return rhs; }
    if (rhs.is_zero()) { return lhs; }

    return ctx.cbh({lhs, rhs},
                   DyadicCachingLayer::metadata().cached_vector_type);
}
algebra::Lie
DyadicCachingLayer::log_signature(const intervals::Interval& domain,
                                  resolution_t resolution,
                                  const algebra::Context& ctx) const
{
    // For now, if the ctx depth is not the same as md depth just do the
    // calculation without caching be smarter about this in the future.
    const auto& md = DyadicCachingLayer::metadata();
    RPY_CHECK(ctx.width() == md.width);
    if (ctx.depth() != md.default_context->depth()) {
        return StreamInterface::log_signature(domain, resolution, ctx);
    }

    auto dyadic_dissection = intervals::to_dyadic_intervals(domain, resolution);

    std::vector<algebra::Lie> lies;
    lies.reserve(dyadic_dissection.size());
    for (const auto& itvl : dyadic_dissection) {
        auto lsig = log_signature(itvl, resolution, ctx);
        if (!lsig.is_zero()) { lies.push_back(lsig); }
    }

    return ctx.cbh(lies, DyadicCachingLayer::metadata().cached_vector_type);
}
algebra::Lie
DyadicCachingLayer::log_signature(const intervals::Interval& interval,
                                  const algebra::Context& ctx) const
{
    return log_signature(interval, metadata().default_resolution, ctx);
}
algebra::FreeTensor
DyadicCachingLayer::signature(const intervals::Interval& interval,
                              const algebra::Context& ctx) const
{
    return signature(interval, metadata().default_resolution, ctx);
}

void DyadicCachingLayer::load_cache() const {
    std::lock_guard<std::recursive_mutex> access(m_compute_lock);
    auto path = get_config().stream_cache_dir() / to_string(m_cache_id);

    if (!exists(path)) {
        return;
    }

//    boost::interprocess::sharable_lock<boost::interprocess::file_lock> fs_access(m_file_lock);

    std::ifstream file_stream(path);
    archives::BinaryInputArchive iar(file_stream);

    iar(m_cache);
}

void DyadicCachingLayer::dump_cache() const {
    std::lock_guard<std::recursive_mutex> access(m_compute_lock);
    if (m_cache.empty()) { return; }

    auto path = get_config().stream_cache_dir();
    if (!exists(path)) {
        create_directories(path);
    }
    path /= to_string(m_cache_id);


//    boost::interprocess::scoped_lock<boost::interprocess::file_lock> fs_access(m_file_lock);

    std::ofstream file_stream(path);
    archives::BinaryOutputArchive oar(file_stream);

    oar(m_cache);
}



RPY_SERIAL_LOAD_FN_IMPL(DyadicCachingLayer) {
    RPY_SERIAL_SERIALIZE_BASE(StreamInterface);
    string tmp;
    RPY_SERIAL_SERIALIZE_NVP("cache_id", tmp);
    m_cache_id = uuids::string_generator()(tmp);
    load_cache();
}

RPY_SERIAL_SAVE_FN_IMPL(DyadicCachingLayer) {
    RPY_SERIAL_SERIALIZE_BASE(StreamInterface);
    RPY_SERIAL_SERIALIZE_NVP("cache_id", to_string(m_cache_id));
    dump_cache();
}

#define RPY_EXPORT_MACRO ROUGHPY_STREAMS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::DyadicCachingLayer
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>


RPY_SERIAL_REGISTER_CLASS(rpy::streams::DyadicCachingLayer)
