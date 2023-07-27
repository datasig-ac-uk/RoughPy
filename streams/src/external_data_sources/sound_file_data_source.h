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

#ifndef ROUGHPY_STREAMS_SRC_EXTERNAL_DATA_SOURCES_SOUND_FILE_DATA_SOURCE_H
#define ROUGHPY_STREAMS_SRC_EXTERNAL_DATA_SOURCES_SOUND_FILE_DATA_SOURCE_H

#include <roughpy/streams/external_data_stream.h>

#include <boost/container/flat_map.hpp>
#include <sndfile.hh>

#include <roughpy/platform/filesystem.h>

namespace rpy {
namespace streams {

class SoundFileDataSource : public ExternalDataStreamSource
{
    param_t m_file_start = 0.0;
    param_t m_time_param_scaling = 1.0;
    SndfileHandle m_handle;

    sf_count_t param_to_frame(param_t param);

    void read_direct_float(scalars::ScalarPointer& ptr, sf_count_t num_frames);
    void read_direct_double(scalars::ScalarPointer& ptr, sf_count_t num_frames);

    void read_convert_raw(scalars::ScalarPointer& ptr, sf_count_t num_frames);

    template <typename T>
    void read_convert(scalars::ScalarPointer& ptr, sf_count_t num_frames)
    {
        const auto num_elements = num_frames * m_handle.channels();
        std::vector<T> buffer(num_elements);
        m_handle.readf(buffer.data(), num_frames);
        ptr.type()->convert_copy(ptr, buffer.data(), num_elements,
                                 scalars::type_id_of<T>());
    }

    void select_and_convert_read2(scalars::ScalarPointer& ptr,
                                  sf_count_t num_frames);
    void select_and_convert_read(scalars::ScalarPointer& ptr,
                                 sf_count_t num_frames);

    template <typename T>
    dimn_t query_impl(scalars::KeyScalarArray& result,
                      const intervals::Interval& interval,
                      const StreamSchema& schema);

public:
    explicit SoundFileDataSource(const url& uri);
    explicit SoundFileDataSource(SndfileHandle&& handle);

    dimn_t query(scalars::KeyScalarArray& result,
                 const intervals::Interval& interval,
                 const StreamSchema& schema
                 ) override;
};

class SoundFileDataSourceFactory : public ExternalDataSourceFactory
{

public:
    void set_width(void* payload, deg_t width) const override;
    void set_depth(void* payload, deg_t depth) const override;
    void set_ctype(void* payload,
                   const scalars::ScalarType* ctype) const override;
    void set_context(void* payload,
                     algebra::context_pointer ctx) const override;
    void set_support(void* payload,
                     intervals::RealInterval support) const override;
    void set_vtype(void* payload, algebra::VectorType vtype) const override;
    void set_resolution(void* payload, resolution_t resolution) const override;
    void set_schema(void* payload, std::shared_ptr<StreamSchema> schema)
            const override;
    void destroy_payload(void*& payload) const override;
    ExternalDataStreamConstructor
    get_constructor(const url& uri) const override;
    Stream construct_stream(void* payload) const override;
};



}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_SRC_EXTERNAL_DATA_SOURCES_SOUND_FILE_DATA_SOURCE_H
