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

#ifndef ROUGHPY_STREAMS_SRC_EXTERNAL_DATA_SOURCES_SOUND_FILE_DATA_SOURCE_H
#define ROUGHPY_STREAMS_SRC_EXTERNAL_DATA_SOURCES_SOUND_FILE_DATA_SOURCE_H

#include "external_data_stream.h"

#include <boost/container/flat_map.hpp>
#include <sndfile.hh>

#include <roughpy/platform/filesystem.h>

namespace rpy {
namespace streams {

class SoundFileDataSource : public ExternalDataStreamSource {
    param_t m_file_start = 0.0;
    param_t m_time_param_scaling = 1.0;
    SndfileHandle m_handle;

    sf_count_t param_to_frame(param_t param);

public:
    scalars::KeyScalarArray query(const intervals::Interval &interval, const scalars::ScalarType *ctype) override;
};

}// namespace streams
}// namespace rpy

#endif//ROUGHPY_STREAMS_SRC_EXTERNAL_DATA_SOURCES_SOUND_FILE_DATA_SOURCE_H
