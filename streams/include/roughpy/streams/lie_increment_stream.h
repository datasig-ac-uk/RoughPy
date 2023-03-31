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

#ifndef ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_
#define ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_

#include "stream_base.h"
#include "dyadic_caching_layer.h"

#include <boost/container/flat_map.hpp>

#include <roughpy/core/helpers.h>
#include <roughpy/scalars/key_scalar_array.h>

namespace rpy { namespace streams {

class ROUGHPY_STREAMS_EXPORT LieIncrementStream : public DyadicCachingLayer {
    scalars::KeyScalarArray m_buffer;
    boost::container::flat_map<param_t, dimn_t> m_mapping;

    using base_t = DyadicCachingLayer;
public:

    LieIncrementStream(
        scalars::KeyScalarArray&& buffer,
        Slice<param_t> indices,
        StreamMetadata md
        );

    bool empty(const intervals::Interval &interval) const noexcept override;

protected:
    algebra::Lie log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const override;

};


}}


#endif // ROUGHPY_STREAMS_LIE_INCREMENT_STREAM_H_
