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

#ifndef ROUGHPY_STREAMS_DYADIC_CACHING_LAYER_H_
#define ROUGHPY_STREAMS_DYADIC_CACHING_LAYER_H_

#include <map>
#include <mutex>

#include <roughpy/platform.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/intervals/dyadic_interval.h>

#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/uuid/uuid.hpp>

#include "stream_base.h"

namespace rpy {
namespace uuids = boost::uuids;

namespace streams {

/**
 * @brief Caching layer utilising a dyadic dissection of the parameter interval.
 *
 * This layer introducing caching for the computation of log signatures by
 * utilising the fact that the signature of a concatenation of paths is the
 * product of signatures (or applying the Campbell-Baker-Hausdorff formula to
 * log signatures). The parameter interval is dissected into dyadic intervals
 * of a resolution and the log signature is computed on all those dyadic
 * intervals that are contained within the requested interval. These are then
 * combined using the Campbell-Baker-Hausdorff formula to give the log signature
 * over the whole interval.
 *
 */
class ROUGHPY_STREAMS_EXPORT DyadicCachingLayer : public StreamInterface
{
    mutable std::map<intervals::DyadicInterval, algebra::Lie> m_cache;
    mutable std::recursive_mutex m_compute_lock;

    uuids::uuid m_cache_id;

public:
    using StreamInterface::StreamInterface;


    DyadicCachingLayer();


    DyadicCachingLayer(const DyadicCachingLayer&) = delete;
    DyadicCachingLayer(DyadicCachingLayer&& other) noexcept;

    DyadicCachingLayer& operator=(const DyadicCachingLayer&) = delete;
    DyadicCachingLayer& operator=(DyadicCachingLayer&& other) noexcept;

    using StreamInterface::log_signature;
    using StreamInterface::signature;

    RPY_NO_DISCARD
    algebra::Lie log_signature(const intervals::Interval& interval,
                               const algebra::Context& ctx) const override;

    RPY_NO_DISCARD
    algebra::Lie log_signature(const intervals::DyadicInterval& interval,
                               resolution_t resolution,
                               const algebra::Context& ctx) const override;

    RPY_NO_DISCARD
    algebra::Lie log_signature(const intervals::Interval& domain,
                               resolution_t resolution,
                               const algebra::Context& ctx) const override;

    RPY_NO_DISCARD
    algebra::FreeTensor signature(const intervals::Interval& interval,
                                  const algebra::Context& ctx) const override;


    const uuids::uuid& cache_id() const noexcept { return m_cache_id; }

    RPY_SERIAL_LOAD_FN();
    RPY_SERIAL_SAVE_FN();

protected:

    void load_cache() const;
    void dump_cache() const;

};

#ifdef RPY_COMPILING_STREAMS
RPY_SERIAL_EXTERN_LOAD_CLS_BUILD(DyadicCachingLayer)
RPY_SERIAL_EXTERN_SAVE_CLS_BUILD(DyadicCachingLayer)
#else
RPY_SERIAL_EXTERN_LOAD_CLS_IMP(DyadicCachingLayer)
RPY_SERIAL_EXTERN_SAVE_CLS_IMP(DyadicCachingLayer)
#endif

}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(::rpy::streams::DyadicCachingLayer,
                            rpy::serial::specialization::member_load_save)

#endif// ROUGHPY_STREAMS_DYADIC_CACHING_LAYER_H_
