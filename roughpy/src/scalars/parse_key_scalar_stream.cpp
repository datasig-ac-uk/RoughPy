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
// Created by sam on 08/08/23.
//

#include "parse_key_scalar_stream.h"
#include "args/numpy.h"
#include "args/strided_copy.h"
#include "args/dlpack_helpers.h"
#include <roughpy/scalars/types.h>

using namespace rpy;
using namespace rpy::python;

using rpy::scalars::KeyScalarArray;
using rpy::scalars::KeyScalarStream;


static inline void buffer_to_stream(
        ParsedKeyScalarStream& result, const py::buffer_info& buf_info,
        PyToBufferOptions& options
);

static inline void dl_to_stream(
        ParsedKeyScalarStream& result, const py::object& dl_object,
        PyToBufferOptions& options
);


void python::parse_key_scalar_stream(ParsedKeyScalarStream& result,
        const py::object& data, rpy::python::PyToBufferOptions& options
)
{


    /*
     * A key-data stream should not represent a single (key-)scalar value,
     * so we only need to deal with the following types:
     *   1) An array/buffer of values,
     *   2) A key-array dict,  // implement later
     *   3) Any other kind of sequential data
     */

    if (py::hasattr(data, "__dlpack__")) {
        dl_to_stream(result, data, options);
    } else if (py::isinstance<py::buffer>(data)) {
        const auto buffer_data = py::reinterpret_borrow<py::buffer>(data);
        buffer_to_stream(result, buffer_data.request(), options);
    } else if (py::isinstance<py::dict>(data)) {
        RPY_THROW(
                std::runtime_error,
                "constructing from a dict of arrays/lists is not yet supported"
        );
    } else if (py::isinstance<py::sequence>(data)) {
        // We always need to make a copy from a Python object
        result.data_buffer = py_to_buffer(data, options);

        /*
         * Now we need to use the options.shape information to construct the
         * stream. How we should interpret the shape values are determined by
         * whether keys are given:
         *   1) If result.data_buffer contains keys, then there are shape.size()
         *      increments, where the ith increment contains shape[i] values.
         *   2) If result.data_buffer does not contain keys, then there are two
         *      cases to handle:
         *      a) if shape.size() != 2, then there are shape.size() increments,
         *         where the size of the ith increment is shape[i];
         *      b) if shape.size() == 2 then
         *         - if shape[0]*shape[1] == data_buffer.size() then there are
         *           shape[0] increments with size shape[1]
         *         - otherwise there are 2 increments of sizes shape[0] and
         *           shape[1].
         *
         * Oof, that's a lot of cases.
         */

        const auto buf_size = result.data_buffer.size();
        if (result.data_buffer.has_keys()) {
            result.data_stream.reserve_size(options.shape.size());

            scalars::ScalarPointer sptr(result.data_buffer);
            const key_type* kptr = result.data_buffer.keys();
            dimn_t check = 0;
            for (auto incr_size : options.shape) {
                result.data_stream.push_back(
                        {sptr, static_cast<dimn_t>(incr_size)}, kptr
                );
                sptr += incr_size;
                kptr += incr_size;
                check += incr_size;
            }

            RPY_CHECK(check == buf_size);
        } else if (options.shape.size() != 2 || options.shape[0] * options.shape[1] != buf_size) {
            result.data_stream.reserve_size(options.shape.size());
            scalars::ScalarPointer sptr(result.data_buffer);
            dimn_t check = 0;

            for (auto incr_size : options.shape) {
                result.data_stream.push_back(
                        {sptr, static_cast<dimn_t>(incr_size)}
                );
                sptr += incr_size;
                check += incr_size;
            }

            RPY_CHECK(check == buf_size);
        } else {
            RPY_DBG_ASSERT(options.shape[0] * options.shape[1] == buf_size);

            const auto num_increments = static_cast<dimn_t>(options.shape[0]);
            const auto incr_size = static_cast<dimn_t>(options.shape[1]);
            result.data_stream.set_elts_per_row(incr_size);

            scalars::ScalarPointer sptr(result.data_buffer);
            for (dimn_t i = 0; i < num_increments; ++i) {
                result.data_stream.push_back(sptr);
                sptr += incr_size;
            }
        }

    } else {
        RPY_THROW(
                std::invalid_argument,
                "could not parse argument to a valid scalar array type"
        );
    }

}

void buffer_to_stream(
        ParsedKeyScalarStream& result, const py::buffer_info& buf_info,
        PyToBufferOptions& options
)
{
    RPY_CHECK(buf_info.ndim <= 2 && buf_info.ndim > 0);

    /*
     * Generally we will want to borrow the data if it is at all possible. There
     * are some caveats though. First, we can only borrow the data if it is
     * contiguous and C-layout. Second, we can only borrow if the data is a
     * simple type and this type matches the requested data type (if set).
     */

    auto type_id = py_buffer_to_type_id(buf_info);

    if (options.type == nullptr) {
        options.type = scalars::ScalarType::for_id(type_id);
    }

    // Imperfect check for whether the chosen data type is the same.
    bool borrow = options.type->id() == type_id;

    // Check if the array is C-contiguous
    auto acc_stride = buf_info.itemsize;
    for (auto dim = buf_info.ndim; dim > 0;) {
        const auto& this_stride = buf_info.strides[--dim];
        RPY_CHECK(this_stride > 0);
        borrow &= this_stride == acc_stride;
        acc_stride *= buf_info.shape[dim];
    }

    if (borrow) {
        if (buf_info.ndim == 1) {
            result.data_stream.set_elts_per_row(buf_info.shape[0]);
            result.data_stream.reserve_size(1);
            result.data_stream.push_back({type_id, buf_info.ptr});
        } else {
            const auto num_increments = static_cast<dimn_t>(buf_info.shape[0]);
            result.data_stream.set_elts_per_row(buf_info.shape[1]);
            result.data_stream.reserve_size(num_increments);

            const auto* ptr = static_cast<const char*>(buf_info.ptr);
            const auto stride = buf_info.strides[0];
            for (dimn_t i = 0; i < num_increments; ++i) {
                result.data_stream.push_back({options.type, ptr});
                ptr += stride;
            }
        }
    } else {
        std::vector<char> tmp(buf_info.size * buf_info.itemsize);
        py::ssize_t tmp_strides[2]{};
        dimn_t tmp_shape[2]{};
        tmp_strides[buf_info.ndim - 1] = buf_info.itemsize;
        bool transposed
                = buf_info.ndim == 2 && buf_info.shape[0] < buf_info.shape[1];
        if (buf_info.ndim == 2) {
            if (transposed) {
                tmp_strides[0] = buf_info.shape[1];
                tmp_shape[0] = buf_info.shape[1];
                tmp_shape[1] = buf_info.shape[0];
            } else {
                tmp_strides[0] = buf_info.shape[0];
                tmp_shape[0] = buf_info.shape[0];
                tmp_shape[1] = buf_info.shape[1];
            }
        }

        stride_copy(
                tmp.data(), buf_info.ptr, buf_info.itemsize, buf_info.ndim,
                buf_info.shape.data(), buf_info.strides.data(), tmp_strides,
                transposed
        );

        // Now that we're C-contiguous, convert_copy into the result.
        result.data_buffer = KeyScalarArray(options.type);
        result.data_buffer.allocate_scalars(buf_info.size);
        options.type->convert_copy(
                result.data_buffer, {type_id, tmp.data()}, buf_info.size
        );

        if (buf_info.ndim == 1) {
            result.data_stream.reserve_size(1);
            result.data_stream.set_elts_per_row(buf_info.size);
            result.data_stream.push_back({options.type, tmp.data()});
        } else {
            // shape[0] increments of size shape[1]
            RPY_DBG_ASSERT(
                    buf_info.shape[0] * buf_info.shape[1] == buf_info.size
            );
            result.data_stream.reserve_size(tmp_shape[0]);
            result.data_stream.set_elts_per_row(tmp_shape[1]);

            scalars::ScalarPointer sptr(result.data_buffer);
            for (dimn_t i = 0; i < tmp_shape[0]; ++i) {
                result.data_stream.push_back(sptr);
                sptr += tmp_shape[1];
            }
        }
    }
}

void dl_to_stream(
        ParsedKeyScalarStream& result, const py::object& dl_object,
        PyToBufferOptions& options
)
{
    /*
     * This is going to work much like the python buffer protocol, except we
     * have to do some extra work to deal with device arrays. Currently, this is
     * not supported, but this might change in the future.
     */

    py::capsule dlpack = dl_object.attr("__dlpack__")();
    auto* dltensor = dlpack.get_pointer<DLManagedTensor>();
    RPY_CHECK(dltensor != nullptr);
    auto& tensor = dltensor->dl_tensor;

    const auto type_id = python::type_id_for_dl_info(tensor.dtype, tensor.device);
    if (options.type == nullptr) {
        options.type = scalars::ScalarType::for_id(type_id);
    }

    if (tensor.ndim == 0
        || tensor.shape[0] == 0
        || (tensor.ndim > 1 && tensor.shape[1] == 0)) {
        return;
    }

    RPY_CHECK(tensor.ndim == 1 || tensor.ndim == 2);
    RPY_CHECK(tensor.dtype.lanes == 1);

    result.data_stream = KeyScalarStream(options.type);
    result.data_buffer = KeyScalarArray(options.type);

    bool borrow = options.type->id() == type_id;

    // Check if the array is C-contiguous
    const auto itemsize = tensor.dtype.bits / 8;
    int64_t size = 1;
    for (int64_t i=0; i<tensor.ndim; ++i) {
        size *= tensor.shape[i];
    }

    borrow &= tensor.strides == nullptr;

    if (borrow) {
        if (tensor.ndim == 1) {
            result.data_stream.set_elts_per_row(tensor.shape[0]);
            result.data_stream.reserve_size(1);
            result.data_stream.push_back({options.type, tensor.data});
        } else {
            const auto num_increments = static_cast<dimn_t>(tensor.shape[0]);
            result.data_stream.set_elts_per_row(tensor.shape[1]);
            result.data_stream.reserve_size(num_increments);

            const auto* ptr = static_cast<const char*>(tensor.data);
            const auto stride = tensor.shape[1] * itemsize;
            for (dimn_t i = 0; i < num_increments; ++i) {
                result.data_stream.push_back({options.type, ptr});
                ptr += stride;
            }
        }
    } else {
        std::vector<char> tmp(size * itemsize);
        py::ssize_t out_strides[2]{};
        py::ssize_t in_strides[2] {};
        py::ssize_t in_shape[2] {};
        dimn_t out_shape[2]{};
        out_strides[tensor.ndim - 1] = itemsize;
        bool transposed = tensor.ndim == 2 && tensor.shape[0] < tensor.shape[1];

        in_shape[0] = tensor.shape[0];
        if (tensor.ndim == 2) {
            if (transposed) {
                out_shape[0] = tensor.shape[1];
                out_shape[1] = tensor.shape[0];
            } else {
                out_shape[0] = tensor.shape[0];
                out_shape[1] = tensor.shape[1];
            }
            out_strides[0] = out_shape[1]*itemsize;

            if (tensor.strides != nullptr) {
                in_strides[0] = tensor.strides[0]*itemsize;
                in_strides[1] = tensor.strides[1]*itemsize;
            } else {
                in_strides[0] = tensor.shape[1]*itemsize;
                in_strides[1] = itemsize;
            }

            in_shape[1] = tensor.shape[1];
        }

        stride_copy(
                tmp.data(), tensor.data, itemsize, tensor.ndim,
                in_shape, in_strides, out_strides,
                transposed
        );

        // Now that we're C-contiguous, convert_copy into the result.
        result.data_buffer = KeyScalarArray(options.type);
        result.data_buffer.allocate_scalars(size);
        options.type->convert_copy(
                result.data_buffer, {type_id, tmp.data()}, size
        );

        if (tensor.ndim == 1) {
            result.data_stream.reserve_size(1);
            result.data_stream.set_elts_per_row(size);
            result.data_stream.push_back({options.type, tmp.data()});
        } else {
            // shape[0] increments of size shape[1]
            RPY_DBG_ASSERT(
                    tensor.shape[0] * tensor.shape[1] == size
            );
            result.data_stream.reserve_size(out_shape[0]);
            result.data_stream.set_elts_per_row(out_shape[1]);

            scalars::ScalarPointer sptr(result.data_buffer);
            for (dimn_t i = 0; i < out_shape[0]; ++i) {
                result.data_stream.push_back(sptr);
                sptr += out_shape[1];
            }
        }
    }
}
