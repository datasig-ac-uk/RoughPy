// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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
// Created by sam on 09/08/23.
//

#include "strided_copy.h"

#include <roughpy/core/types.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

#include <cstring>
#include <functional>
#include <numeric>

#include <boost/container/small_vector.hpp>

using namespace rpy;

void rpy::python::stride_copy(
        void* dst,
        const void* src,
        const py::ssize_t itemsize,
        const py::ssize_t ndim,
        const py::ssize_t* shape_in,
        const py::ssize_t* strides_in,
        const py::ssize_t* strides_out,
        bool transpose
) noexcept
{
    RPY_DBG_ASSERT(ndim == 1 || ndim == 2);

    auto* dptr = static_cast<char*>(dst);
    const auto* sptr = static_cast<const char*>(src);

    if (ndim == 1) {
        if (strides_in[0] == itemsize) {
            std::memcpy(dptr, sptr, shape_in[0] * itemsize);
        } else {
            for (py::ssize_t i = 0; i < shape_in[0]; ++i) {
                std::memcpy(
                        dptr + i * strides_out[0],
                        sptr + i * strides_in[0],
                        itemsize
                );
            }
        }
    } else if (transpose) {
        for (py::ssize_t i = 0; i < shape_in[1]; ++i) {
            for (py::ssize_t j = 0; j < shape_in[0]; ++j) {
                std::memcpy(
                        dptr + j * strides_out[0] + i * strides_out[1],
                        sptr + i * strides_in[1] + j * strides_in[0],
                        itemsize
                );
            }
        }
    } else {
        for (py::ssize_t i = 0; i < shape_in[0]; ++i) {
            for (py::ssize_t j = 0; j < shape_in[1]; ++j) {
                std::memcpy(
                        dptr + i * strides_out[0] + j * strides_out[1],
                        sptr + i * strides_in[0] + j * strides_in[1],
                        itemsize
                );
            }
        }
    }
}

namespace {

int32_t get_compressed_dims(
        int32_t ndim,
        int32_t bytes,
        const idimn_t* shape,
        const idimn_t* strides
) noexcept
{
    if (strides == nullptr) { return ndim; }

    idimn_t compressed_stride = bytes;

    for (int32_t i = 1; i <= ndim; ++i) {
        if (strides[ndim - i] != compressed_stride) { return i - 1; }
        compressed_stride *= shape[ndim - i];
    }

    return ndim;
}

}// namespace

void rpy::python::stride_copy(
        rpy::scalars::ScalarArray& out,
        const rpy::scalars::ScalarArray& in,
        const int32_t ndim,
        const idimn_t* shape,
        const idimn_t* strides
)
{
    if (ndim == 0) { return; }
    const auto info = in.type_info();

    auto compressed_ndim
            = get_compressed_dims(ndim, info.bytes, shape, strides);

    RPY_DBG_ASSERT(out.type());
    auto tp = *out.type();

    if (compressed_ndim == ndim) {
        // Fully compressed already, just invoke copy-convert
        tp->convert_copy(out, in);
    } else {
        const auto uncompressed = ndim - compressed_ndim;
        const byte* src = reinterpret_cast<const byte*>(in.pointer());
        boost::container::small_vector<idimn_t, 2> index(uncompressed);

        auto src_offset = [&strides, &uncompressed, &index]() {
            dimn_t offset = 0;
            for (int32_t i = 0; i < uncompressed; ++i) {
                offset += strides[i] * index[i];
            }
            return offset;
        };

        auto dst_offset = [&shape, &uncompressed, &index]() {
            dimn_t offset = index[uncompressed-1];
            for (int32_t i = 1; i < uncompressed; ++i) {
                offset += shape[i] * index[i-1];
            }
            return offset;
        };

        auto advance_index = [&shape, &index]() {
            auto dim = index.size() - 1;
            for (auto it = index.rbegin(); it != index.rend(); ++it, --dim) {
                if (++(*it) >= shape[dim] && dim > 0) {
                    *it = 0;
                } else {
                    break;
                }
            }
        };

        auto compression_size = std::accumulate(
                shape + uncompressed,
                shape + ndim,
                static_cast<idimn_t>(1),
                std::multiplies<>()
        );

        for (; index[0] < shape[0]; advance_index()) {
            auto offset = dst_offset();
            auto tmp_out = out[{offset, offset + compression_size}];
            scalars::ScalarArray tmp_in(
                    info,
                    src + src_offset(),
                    compression_size
            );
            tp->convert_copy(tmp_out, tmp_in);
        }
    }
}
