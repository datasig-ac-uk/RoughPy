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
// Created by user on 10/03/23.
//

#include <roughpy/streams/lie_increment_stream.h>

using namespace rpy;
using namespace rpy::streams;

LieIncrementStream::LieIncrementStream(scalars::KeyScalarArray&& buffer,
                                       Slice<param_t> indices,
                                       StreamMetadata metadata)
    : base_t(std::move(metadata)), m_buffer(std::move(buffer))
{
    const auto& md = this->metadata();
    for (dimn_t i = 0; i < indices.size(); ++i) {
        m_mapping[indices[i]] = i * md.width;
    }

    //    std::cerr << m_mapping.begin()->first << ' ' <<
    //    (--m_mapping.end())->first << '\n';
}

algebra::Lie
LieIncrementStream::log_signature_impl(const intervals::Interval& interval,
                                       const algebra::Context& ctx) const
{

    const auto& md = metadata();
    //    if (empty(interval)) {
    //        return ctx.zero_lie(md.cached_vector_type);
    //    }

    rpy::algebra::SignatureData data{scalars::ScalarStream(ctx.ctype()),
                                     {},
                                     md.cached_vector_type};

    if (m_mapping.size() == 1) {
        data.data_stream.set_elts_per_row(m_buffer.size());
    } else if (m_mapping.size() > 1) {
        auto row1 = (++m_mapping.begin())->second;
        auto row0 = m_mapping.begin()->second;
        data.data_stream.set_elts_per_row(row1 - row0);
    }

    auto begin = (interval.type() == intervals::IntervalType::Opencl)
            ? m_mapping.upper_bound(interval.inf())
            : m_mapping.lower_bound(interval.inf());

    auto end = (interval.type() == intervals::IntervalType::Opencl)
            ? m_mapping.upper_bound(interval.sup())
            : m_mapping.lower_bound(interval.sup());

    if (begin == end) { return ctx.zero_lie(md.cached_vector_type); }

    data.data_stream.reserve_size(end - begin);

    for (auto it1 = begin, it = it1++; it1 != end; ++it, ++it1) {
        data.data_stream.push_back(
                {m_buffer[it->second].to_pointer(), it1->second - it->second});
    }
    // Case it = it1 - 1 and it1 == end
    --end;
    data.data_stream.push_back({m_buffer[end->second].to_pointer(),
                                m_buffer.size() - end->second});

    if (m_buffer.keys() != nullptr) {
        data.key_stream.reserve(end - begin);
        ++end;
        for (auto it = begin; it != end; ++it) {
            data.key_stream.push_back(m_buffer.keys() + it->second);
        }
    }

    RPY_CHECK(ctx.width() == md.width);
    //    assert(ctx.depth() == md.depth);

    return ctx.log_signature(data);
}
bool LieIncrementStream::empty(
        const intervals::Interval& interval) const noexcept
{
    //    std::cerr << "Checking " << interval;
    //    for (auto& item : m_mapping) {
    //        if (item.first >= interval.inf() && item.first < interval.sup()) {
    //            std::cerr << ' ' << item.first << ", " << item.second << ';';
    //        }
    //    }
    auto begin = (interval.type() == intervals::IntervalType::Opencl)
            ? m_mapping.upper_bound(interval.inf())
            : m_mapping.lower_bound(interval.inf());

    auto end = (interval.type() == intervals::IntervalType::Opencl)
            ? m_mapping.upper_bound(interval.sup())
            : m_mapping.lower_bound(interval.sup());

    //    std::cerr << ' ' << (begin == end) << '\n';
    return begin == end;
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::LieIncrementStream

#include <roughpy/platform/serialization_instantiations.inl>
