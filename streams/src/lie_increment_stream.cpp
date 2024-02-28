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

#include <cereal/types/concepts/pair_associative_container.hpp>

#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <vector>

using namespace rpy;
using namespace rpy::streams;

LieIncrementStream::LieIncrementStream(
        const scalars::KeyScalarArray& buffer, Slice<param_t> indices,
        StreamMetadata metadata, std::shared_ptr<StreamSchema> schema
)
    : base_t(std::move(metadata), std::move(schema))
{
    using scalars::Scalar;

    const auto& md = this->metadata();
    const auto& ctx = *md.default_context;

    const auto& sch = this->schema();
    const auto* param = sch.parametrization();
    const bool param_needs_adding = param != nullptr && param->needs_adding();
    const key_type param_slot
            = (param_needs_adding) ? sch.time_channel_to_lie_key() : 0;

    m_data.reserve(indices.size());

    if (!buffer.is_null()) {
        if (buffer.has_keys()) {
            /*
             * The data is sparse, so we need to do some careful checking to make
             * sure we pick out individual increments.
             * TODO: Need key-scalar stream to implement this properly.
             */
            RPY_THROW(
                    std::runtime_error,
                    "creating a Lie increment stream with sparse data is not "
                    "currently supported"
            );

        } else {
            /*
             * The data is dense. The only tricky part for this case is dealing with
             * adding the "time" channel if it is given.
             *
             * Until we construct the relevant support mechanisms, we assume that
             * the provided increments have degree 1.
             * TODO: Add support for key-scalar streams.
             */
            const auto info = buffer.type_info();
            const auto width = sch.width_without_param();

            const char* dptr = buffer.as_slice<const char>().data();
            const auto stride = info.bytes * width;
            param_t previous_param = 0.0;
            for (auto index : indices) {

                algebra::VectorConstructionData cdata{
                        scalars::KeyScalarArray(*buffer.type(), dptr, width),
                        md.cached_vector_type};

                auto [it, inserted]
                        = m_data.try_emplace(index, ctx.construct_lie(cdata));

                if (inserted && param_needs_adding) {
                    /*
                     * We've inserted a new element, so we should now add the param
                     * value if it is needed.
                     */
                    it->second[param_slot] = Scalar(index - previous_param);
                }
                previous_param = index;
                dptr += stride;
            }
        }
    }
}

LieIncrementStream::LieIncrementStream(
        const scalars::KeyScalarStream& ks_stream, Slice<param_t> indices,
        StreamMetadata mdarg, std::shared_ptr<StreamSchema> schema_arg
)
    : DyadicCachingLayer(std::move(mdarg), std::move(schema_arg))
{
    using scalars::Scalar;
    RPY_CHECK(indices.size() == ks_stream.row_count());

    const auto& md = this->metadata();
    const auto& ctx = *md.default_context;

    const auto& sch = this->schema();
    const auto* param = sch.parametrization();
    const bool param_needs_adding = param != nullptr && param->needs_adding();
    const key_type param_slot
            = (param_needs_adding) ? sch.time_channel_to_lie_key() : 0;

    m_data.reserve(indices.size());
    param_t previous_param = 0.0;

    for (dimn_t i=0; i<indices.size(); ++i) {
        const auto& index = indices[i];

        algebra::VectorConstructionData cdata {
                ks_stream[i],
                md.cached_vector_type
        };

        auto [it, inserted]
                = m_data.try_emplace(index, ctx.construct_lie(cdata));


        if (inserted && param_needs_adding) {
            /*
             * We've inserted a new element, so we should now add the param
             * value if it is needed.
             */
            it->second[param_slot] = index - previous_param;
        }
        previous_param = index;

    }
}

algebra::Lie LieIncrementStream::log_signature_impl(
        const intervals::Interval& interval, const algebra::Context& ctx
) const
{

    const auto& md = metadata();

    auto begin = (interval.type() == intervals::IntervalType::Opencl)
            ? m_data.upper_bound(interval.inf())
            : m_data.lower_bound(interval.inf());

    auto end = (interval.type() == intervals::IntervalType::Opencl)
            ? m_data.upper_bound(interval.sup())
            : m_data.lower_bound(interval.sup());

    if (begin == end) { return ctx.zero_lie(md.cached_vector_type); }

    std::vector<const Lie*> lies;
    lies.reserve(static_cast<dimn_t>(end - begin));

    for (auto it = begin; it != end; ++it) {
        lies.push_back(&it->second);
    }

    return ctx.cbh(lies, md.cached_vector_type);
}
bool LieIncrementStream::empty(const intervals::Interval& interval
) const noexcept
{
    auto begin = (interval.type() == intervals::IntervalType::Opencl)
            ? m_data.upper_bound(interval.inf())
            : m_data.lower_bound(interval.inf());

    auto end = (interval.type() == intervals::IntervalType::Opencl)
            ? m_data.upper_bound(interval.sup())
            : m_data.lower_bound(interval.sup());

    return begin == end;
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
#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::LieIncrementStream

#include <roughpy/platform/serialization_instantiations.inl>

RPY_SERIAL_REGISTER_CLASS(rpy::streams::LieIncrementStream)
