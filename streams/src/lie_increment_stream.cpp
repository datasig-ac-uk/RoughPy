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

#include <roughpy/streams/lie_increment_stream.h>
#include "roughpy/core/check.h"             // for throw_exception, RPY_CHECK
#include "roughpy/core/ranges.h"

#include <cereal/types/concepts/pair_associative_container.hpp>

#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <vector>

using namespace rpy;
using namespace rpy::streams;




LieIncrementStream::LieIncrementStream(std::vector<pair<param_t, Lie> >&& data,
                                       std::shared_ptr<const StreamMetadata> md)
    : p_metadata(std::move(md))
{
    m_data.reserve(data.size());
    for (auto&& [param, lie] : data | views::move) {
        m_data.emplace(param, std::move(lie));
    }

}

algebra::Lie LieIncrementStream::log_signature_impl(
    const intervals::DyadicInterval& interval,
    resolution_t resolution,
    const algebra::Context& ctx
) const
{
    const auto& md = metadata();

    auto begin = (interval.type() == intervals::IntervalType::Opencl)
            ? m_data.upper_bound(interval.inf())
            : m_data.lower_bound(interval.inf());

    auto end = (interval.type() == intervals::IntervalType::Opencl)
            ? m_data.upper_bound(interval.sup())
            : m_data.lower_bound(interval.sup());

    if (begin == end) { return zero_lie(); }

    std::vector<const Lie*> lies;
    lies.reserve(static_cast<dimn_t>(end - begin));

    for (auto it = begin; it != end; ++it) { lies.push_back(&it->second); }

    return ctx.cbh(lies, algebra::VectorType::Dense);
}



#define RPY_EXPORT_MACRO ROUGHPY_STREAMS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::LieIncrementStream

#include <roughpy/platform/serialization_instantiations.inl>

RPY_SERIAL_REGISTER_CLASS(rpy::streams::LieIncrementStream)