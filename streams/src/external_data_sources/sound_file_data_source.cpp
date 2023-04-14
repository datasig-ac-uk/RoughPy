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

#include "sound_file_data_source.h"

#include <roughpy/platform/filesystem.h>

#include <cmath>

using namespace rpy;
using namespace rpy::streams;


sf_count_t SoundFileDataSource::param_to_frame(param_t param) {
    assert(static_cast<bool>(m_handle));
    assert(param >= 0.0);
    auto sample_rate = static_cast<param_t>(m_handle.samplerate());

    auto seconds = (param - m_file_start) * m_time_param_scaling;

    return static_cast<sf_count_t>(std::ceil(seconds * sample_rate));
}

void SoundFileDataSource::read_direct_float(scalars::ScalarPointer &ptr, sf_count_t num_frames) {
    m_handle.readf(ptr.raw_cast<float*>(), num_frames);
}
void SoundFileDataSource::read_direct_double(scalars::ScalarPointer &ptr, sf_count_t num_frames) {
    m_handle.readf(ptr.raw_cast<double*>(), num_frames);
}
void SoundFileDataSource::read_convert_raw(scalars::ScalarPointer &ptr, sf_count_t num_frames) {
    const auto num_elements = num_frames*m_handle.channels();
    std::vector<int8_t> buffer(num_elements);
    m_handle.readRaw(buffer.data(), num_elements);
    ptr.type()->convert_copy(ptr, buffer.data(), num_elements, "i8");
}

void SoundFileDataSource::select_and_convert_read2(scalars::ScalarPointer &ptr, sf_count_t num_frames) {
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

void SoundFileDataSource::select_and_convert_read(scalars::ScalarPointer &ptr, sf_count_t num_frames) {

    switch (m_handle.format() & SF_FORMAT_SUBMASK) {
        case SF_FORMAT_PCM_16: read_convert<int16_t>(ptr, num_frames); break;
        case SF_FORMAT_PCM_32: read_convert<int32_t>(ptr, num_frames); break;
        case SF_FORMAT_FLOAT: read_convert<float>(ptr, num_frames); break;
        case SF_FORMAT_DOUBLE: read_convert<double>(ptr, num_frames); break;
        default: select_and_convert_read2(ptr, num_frames);
    }
}

SoundFileDataSource::SoundFileDataSource(const url &uri)
    : m_handle(uri.path().c_str())
{
}

dimn_t SoundFileDataSource::query(scalars::KeyScalarArray &result,
                                const intervals::Interval &interval) {
    const auto *ctype = result.type();

    auto frame_begin = param_to_frame(interval.inf());
    auto frame_end = param_to_frame(interval.sup());

    assert(frame_begin >= 0 && frame_begin <= frame_end && frame_end <= m_handle.frames());

    auto frame_count = frame_end - frame_begin;
    auto seek_pos = m_handle.seek(frame_begin, SEEK_SET);
    if (seek_pos == -1) {
        throw std::runtime_error("invalid seek");
    }

    result.allocate_scalars(frame_count * m_handle.channels());

    const auto &sinfo = ctype->info();

    if (sinfo.device.device_type == scalars::ScalarDeviceType::CPU) {
        if (sinfo.basic_info.code == scalars::ScalarTypeCode::Float) {
            if (sinfo.basic_info.bits == 32) {
                read_direct_float(result, frame_count);
            } else if (sinfo.basic_info.bits == 64) {
                read_direct_double(result, frame_count);
            } else {
                select_and_convert_read(result, frame_count);
            }
        } else {
            select_and_convert_read(result, frame_count);
        }
    } else {
        if (sinfo.basic_info.code == scalars::ScalarTypeCode::Float) {
            if (sinfo.basic_info.bits == 32) {
                read_convert<float>(result, frame_count);
            } else if (sinfo.basic_info.bits == 64) {
                read_convert<double>(result, frame_count);
            } else {
                select_and_convert_read(result, frame_count);
            }
        } else {
            select_and_convert_read(result, frame_count);
        }
    }

    return static_cast<dimn_t>(frame_count);
}
Stream SoundFileDataSourceFactory::construct_stream(const url &uri, StreamMetadata md) const {
    return Stream(ExternalDataStream(SoundFileDataSource(uri), std::move(md)));
}
bool SoundFileDataSourceFactory::supports(const url &uri) const {
    if (uri.has_scheme() && uri.scheme() == "file") {
        fs::path path(uri.path());

        if (exists(path) && is_regular_file(path)) {
            SndfileHandle sf(path.c_str());
            if (sf.error() == 0) {
                return true;
            }
        }
    }

    return false;
}



static RegisterExternalDataSourceFactoryHelper<SoundFileDataSourceFactory> s_register_helper;
