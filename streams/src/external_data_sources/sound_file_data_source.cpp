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

#include "sound_file_data_source.h"

#include <roughpy/platform/filesystem.h>
#include <roughpy/scalars/owned_scalar_array.h>

#include <cmath>

using namespace rpy;
using namespace rpy::streams;

template <typename T>
static scalars::ScalarPointer to_sp(T* ptr)
{
    return scalars::ScalarPointer(scalars::ScalarType::of<T>(), ptr);
}

template <>
scalars::ScalarPointer to_sp<int32_t>(int32_t* ptr)
{
    return scalars::ScalarPointer(ptr);
}

template <>
scalars::ScalarPointer to_sp<int16_t>(int16_t* ptr)
{
    return scalars::ScalarPointer(ptr);
}




sf_count_t SoundFileDataSource::param_to_frame(param_t param)
{
    RPY_DBG_ASSERT(static_cast<bool>(m_handle));
    RPY_DBG_ASSERT(param >= 0.0);
    auto sample_rate = static_cast<param_t>(m_handle.samplerate());

    auto seconds = (param - m_file_start) * m_time_param_scaling;

    return static_cast<sf_count_t>(std::ceil(seconds * sample_rate));
}

void SoundFileDataSource::read_direct_float(scalars::ScalarPointer& ptr,
                                            sf_count_t num_frames)
{
    m_handle.readf(ptr.raw_cast<float*>(), num_frames);
}
void SoundFileDataSource::read_direct_double(scalars::ScalarPointer& ptr,
                                             sf_count_t num_frames)
{
    m_handle.readf(ptr.raw_cast<double*>(), num_frames);
}
void SoundFileDataSource::read_convert_raw(scalars::ScalarPointer& ptr,
                                           sf_count_t num_frames)
{
    const auto num_elements = num_frames * m_handle.channels();
    std::vector<int8_t> buffer(num_elements);
    m_handle.readRaw(buffer.data(), num_elements);
    ptr.type()->convert_copy(ptr, buffer.data(), num_elements, "i8");
}

void SoundFileDataSource::select_and_convert_read2(scalars::ScalarPointer& ptr,
                                                   sf_count_t num_frames)
{
    const auto* ctype = ptr.type();

    const auto& info = ctype->info();
    if (info.basic_info.code == scalars::ScalarTypeCode::Float) {
        // float and double handle earlier, handle smaller and larger.
        if (info.basic_info.bits < 32) {
            read_convert<float>(ptr, num_frames);
        } else {
            read_convert<double>(ptr, num_frames);
        }
    } else {
        read_convert<double>(ptr, num_frames);
    }
}

void SoundFileDataSource::select_and_convert_read(scalars::ScalarPointer& ptr,
                                                  sf_count_t num_frames)
{

    switch (m_handle.format() & SF_FORMAT_SUBMASK) {
        case SF_FORMAT_PCM_16: read_convert<int16_t>(ptr, num_frames); break;
        case SF_FORMAT_PCM_32: read_convert<int32_t>(ptr, num_frames); break;
        case SF_FORMAT_FLOAT: read_convert<float>(ptr, num_frames); break;
        case SF_FORMAT_DOUBLE: read_convert<double>(ptr, num_frames); break;
        default: select_and_convert_read2(ptr, num_frames);
    }
}


template <typename T>
dimn_t SoundFileDataSource::query_impl(
        scalars::KeyScalarArray& result, const intervals::Interval& interval,
        const StreamSchema& schema
)
{
    /*
     * Soooo, we need to deal with the schema in the room. We need to difference
     * any channel marked as values, and discard the first value if necessary.
     * Now things get complicated quickly if something other than this fairly
     * simple transform is applied. We'll pretend that doesn't happen for now.
     *
     */
    const auto* ctype = result.type();


    // At some point, we might want to avoid skipping the first readings.
//    bool has_values = true;

    auto width = schema.width();

    std::vector<T> working(width);
    std::vector<T> previous(m_handle.channels());
    std::vector<T> current(m_handle.channels());


    auto frame_begin = param_to_frame(interval.inf());
    auto frame_end = param_to_frame(interval.sup());

    if (frame_begin == frame_end) {
        // TODO: Check that this is the correct handling
        return 0;
    }

    // If we're at the beginning of time, skip the first reading
    if (frame_begin == 0) {
        ++frame_begin;
    }

    auto seek_pos = m_handle.seek(frame_begin-1, SEEK_SET);
    m_handle.readf(previous.data(), 1);

    RPY_CHECK(frame_begin > 0 && frame_begin <= frame_end
                  && frame_end <= m_handle.frames());

    auto frame_count = frame_end - frame_begin;

    if (seek_pos == -1) { RPY_THROW(std::runtime_error, "invalid seek"); }

    result.allocate_scalars(frame_count * width);

    auto stride = ctype->itemsize() * width;
    char* write_ptr = result.raw_cast<char*>();

    for (sf_count_t row_idx=0; row_idx < frame_count; ++row_idx) {
        /*
         * Process is as follows:
         *      1) Read the data into current.
         *      2) For each channel, write then current value (for increments)
         *         or the differenced value (for values) into the corresponding
         *         slot of working.
         *      3) Swap previous and current - next time around we'll overwrite
         *         the old previous values.
         *      4) convert-copy the working array into the result array.
         *      5) Repeat.
         */

        m_handle.readf(current.data(), 1);
        dimn_t i=0;
        // I really wish C++ had enumerate iterators!
        for (const auto& [_, chan] : schema) {
            auto out_idx = schema.channel_to_stream_dim(i);

            if (chan.type() == ChannelType::Value) {
                working[out_idx] = current[i] - previous[i];
            } else {
                working[out_idx] = current[i];
            }

            ++i;
        }
        std::swap(current, previous);

        ctype->convert_copy(scalars::ScalarPointer(ctype, write_ptr),
                            to_sp(working.data()), width);

        write_ptr += stride;
    }

    return static_cast<dimn_t>(frame_count);
}



SoundFileDataSource::SoundFileDataSource(const url& uri)
    : m_handle(uri.path().c_str())
{}

SoundFileDataSource::SoundFileDataSource(SndfileHandle&& handle)
    : m_handle(std::move(handle))
{}

dimn_t SoundFileDataSource::query(scalars::KeyScalarArray& result,
                                  const intervals::Interval& interval,
                                  const StreamSchema& schema)
{
    /*
     * The actual implementation is dispatched to an appropriate
     * working type via query_impl. For instance, if the channel values are
     * held as 16/32-bit integers, floats, or doubles, we use the appropriate
     * integer/float type. Otherwise, we boost up to either float or double,
     * depending on whether the stream scalar types are floats or doubles.
     * This should dramatically simplify the code.
     */
    auto format = m_handle.format() & SF_FORMAT_SUBMASK;
//
//    switch (format) {
//        case SF_FORMAT_PCM_16:
//            return query_impl<int16_t>(result, interval, schema);
//        case SF_FORMAT_PCM_32:
//            return query_impl<int32_t>(result, interval, schema);
//        case SF_FORMAT_FLOAT:
//            return query_impl<float>(result, interval, schema);
//        case SF_FORMAT_DOUBLE:
//            return query_impl<double>(result, interval, schema);
//        default:
//            break;
//    }
//
    const auto* ctype = result.type();
    /*
     * If we made it here, we need to decide, based on ctype whether to use
     * floats or doubles for the working type.
     */
    auto info = ctype->info();
    switch (info.basic_info.code) {
        case scalars::ScalarTypeCode::Float:
            if (info.basic_info.bits > 16) {
                return query_impl<double>(result, interval, schema);
            } else {
                return query_impl<float>(result, interval, schema);
            }
        case scalars::ScalarTypeCode::Int:
        case scalars::ScalarTypeCode::UInt:
        case scalars::ScalarTypeCode::BFloat:
        case scalars::ScalarTypeCode::OpaqueHandle:
            return query_impl<double>(result, interval, schema);
        case scalars::ScalarTypeCode::Complex:
        case scalars::ScalarTypeCode::Bool:
            RPY_THROW(std::runtime_error, "no conversion to complex or bool "
                                          "types");
    }

    RPY_UNREACHABLE_RETURN(0);
}

namespace {

struct Payload {
    SndfileHandle handle;
    optional<deg_t> width;
    optional<deg_t> depth;
    const scalars::ScalarType* ctype;
    algebra::context_pointer ctx;
    optional<intervals::RealInterval> support;
    optional<algebra::VectorType> vtype;
    optional<resolution_t> resolution;
    std::shared_ptr<StreamSchema> schema;
};

}// namespace

ExternalDataStreamConstructor
SoundFileDataSourceFactory::get_constructor(const url& uri) const
{
    ExternalDataStreamConstructor ctor;
    if (!uri.has_scheme() || uri.scheme_id() == URIScheme::file) {
        fs::path path(uri.path());
#ifdef RPY_PLATFORM_WINDOWS
        path.make_preferred();
#endif
        if (exists(path) && is_regular_file(path)) {
            auto* payload = new Payload{SndfileHandle(path.c_str())};

            if (payload->handle.error() != 0) {
                delete payload;
                return ctor;
            }

            ctor = ExternalDataStreamConstructor(this, payload);
        }
    }

    return ctor;
}
Stream SoundFileDataSourceFactory::construct_stream(void* payload) const
{
    auto* pl = reinterpret_cast<Payload*>(payload);

    StreamMetadata meta{0,       intervals::RealInterval(0, 1), nullptr,
                        nullptr, algebra::VectorType::Dense,    10};

    if (pl->width) {
        auto width = *pl->width;
        if (width != pl->handle.channels()) {
            RPY_THROW(std::invalid_argument,
                    "requested width does not match number of channels");
        }
        meta.width = width;
    } else {
        meta.width = static_cast<deg_t>(pl->handle.channels());
    }

    if (pl->ctype != nullptr) {
        meta.data_scalar_type = pl->ctype;
    } else {
        switch (pl->handle.format() & SF_FORMAT_SUBMASK) {
            case SF_FORMAT_PCM_S8:
                meta.data_scalar_type = scalars::ScalarType::for_id("i8");
                break;
            case SF_FORMAT_PCM_16:
                meta.data_scalar_type = scalars::ScalarType::for_id("i16");
                break;
            case SF_FORMAT_PCM_32:
                meta.data_scalar_type = scalars::ScalarType::for_id("i32");
                break;
            case SF_FORMAT_FLOAT:
                meta.data_scalar_type = scalars::ScalarType::of<float>();
                break;
            default:
            case SF_FORMAT_DOUBLE:
                meta.data_scalar_type = scalars::ScalarType::of<double>();
                break;
        }
    }

    if (pl->ctx != nullptr) {
        // The width check should have already caught a mismatch between
        // width and channels
        meta.default_context = pl->ctx;
    } else {
        if (meta.width != 0 && pl->depth && meta.data_scalar_type != nullptr) {
            meta.default_context = algebra::get_context(
                    meta.width, *pl->depth, meta.data_scalar_type, {});
        } else {
            RPY_THROW(std::invalid_argument,
                    "insufficient information to get context");
        }
    }

    if (pl->vtype) { meta.cached_vector_type = *pl->vtype; }
    if (pl->resolution) { meta.default_resolution = *pl->resolution; }

    if (pl->support) {
        meta.effective_support = *pl->support;
    } else {
        auto length = static_cast<param_t>(pl->handle.frames())
                / pl->handle.samplerate();
        meta.effective_support = intervals::RealInterval(0.0, length);
    }

    if (!pl->schema) {
        pl->schema = std::make_shared<StreamSchema>(meta.width);
    }

    // Let the library handle normalisation
    pl->handle.command(SFC_SET_NORM_FLOAT, nullptr, SF_TRUE);
    pl->handle.command(SFC_SET_NORM_DOUBLE, nullptr, SF_TRUE);

    ExternalDataStream inner(SoundFileDataSource(std::move(pl->handle)),
                             std::move(meta),
                             std::move(pl->schema));

    destroy_payload(payload);

    return Stream(std::move(inner));
}

void SoundFileDataSourceFactory::destroy_payload(void*& payload) const
{
    delete reinterpret_cast<Payload*>(payload);
    payload = nullptr;
}

void SoundFileDataSourceFactory::set_width(void* payload, deg_t width) const
{
    reinterpret_cast<Payload*>(payload)->width = width;
}
void SoundFileDataSourceFactory::set_depth(void* payload, deg_t depth) const
{
    reinterpret_cast<Payload*>(payload)->depth = depth;
}
void SoundFileDataSourceFactory::set_ctype(
        void* payload, const scalars::ScalarType* ctype) const
{
    reinterpret_cast<Payload*>(payload)->ctype = ctype;
}
void SoundFileDataSourceFactory::set_context(void* payload,
                                             algebra::context_pointer ctx) const
{
    reinterpret_cast<Payload*>(payload)->ctx = std::move(ctx);
}
void SoundFileDataSourceFactory::set_support(
        void* payload, intervals::RealInterval support) const
{
    reinterpret_cast<Payload*>(payload)->support = support;
}
void SoundFileDataSourceFactory::set_vtype(void* payload,
                                           algebra::VectorType vtype) const
{
    reinterpret_cast<Payload*>(payload)->vtype = vtype;
}
void SoundFileDataSourceFactory::set_resolution(void* payload,
                                                resolution_t resolution) const
{
    reinterpret_cast<Payload*>(payload)->resolution = resolution;
}
void SoundFileDataSourceFactory::set_schema(
        void* payload, std::shared_ptr<StreamSchema> schema
) const
{
    reinterpret_cast<Payload*>(payload)->schema = std::move(schema);
}
