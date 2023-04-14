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

#include <cmath>


using namespace rpy;
using namespace rpy::streams;

sf_count_t SoundFileDataSource::param_to_frame(param_t param) {
    assert(static_cast<bool>(m_handle));
    assert(param >= 0.0);
    auto sample_rate = static_cast<param_t>(m_handle.samplerate());

    auto seconds = (param - m_file_start)*m_time_param_scaling;

    return static_cast<sf_count_t>(std::ceil(seconds*sample_rate));
}

scalars::KeyScalarArray SoundFileDataSource::query(const intervals::Interval &interval, const scalars::ScalarType *ctype) {
    auto frame_begin = param_to_frame(interval.inf());
    auto frame_end = param_to_frame(interval.sup());

    assert(frame_begin >= 0 && frame_begin <= frame_end && frame_end <= m_handle.frames());

    auto frame_count = frame_end - frame_begin;
    auto seek_pos = m_handle.seek(frame_begin, SEEK_SET);
    if (seek_pos == -1) {
        throw std::runtime_error("invalid seek");
    }



}
