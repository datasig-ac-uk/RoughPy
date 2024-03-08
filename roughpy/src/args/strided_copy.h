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

#ifndef ROUGHPY_STRIDED_COPY_H
#define ROUGHPY_STRIDED_COPY_H

#include "roughpy_module.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/scalars/scalars_fwd.h>

namespace rpy {
namespace python {

/**
 * @brief Copies data from source to destination using strides.
 *
 * This function copies data from the source array to the destination array
 * using given strides. It supports 1-dimensional and 2-dimensional arrays.
 *
 * @param dst Pointer to the destination array.
 * @param src Pointer to the source array.
 * @param itemsize Size of each item in bytes.
 * @param ndim Number of dimensions in the arrays (1 or 2).
 * @param shape_in Array containing the shape of the input array.
 * @param strides_in Array containing the strides of the input array.
 * @param strides_out Array containing the strides of the output array.
 * @param transpose Flag indicating whether to transpose the input array.
 *
 * @return None.
 *
 * @note The function assumes that the input arrays are valid and have
 * compatible dimensions and strides.
 */
void stride_copy(
        void* RPY_RESTRICT dst,
        const void* RPY_RESTRICT src,
        const py::ssize_t itemsize,
        const py::ssize_t ndim,
        const py::ssize_t* shape_in,
        const py::ssize_t* strides_in,
        const py::ssize_t* strides_out,
        bool transpose
) noexcept;

/**
 * @file
 * @brief Check if a slice is C-contiguous
 */
template <typename T>
inline bool is_C_contiguous(
        Slice<const T> strides,
        Slice<const T> shape,
        optional<int32_t> itemsize
) noexcept
{
    if (strides.empty()) { return true; }
    T compressed_size = (itemsize) ? *itemsize : 1;
    for (auto i = strides.size(); i > 0;) {
        if (strides[--i] != compressed_size) { return false; }
        compressed_size *= shape[i];
    }
    return true;
}

/**
 * @brief Copies data from one ScalarArray to another, with optional compression
 * of dimensions.
 *
 * This function copies data from the input ScalarArray `in` to the output
 * ScalarArray `out`. The `ndim` parameter specifies the number of dimensions in
 * the arrays. The `shape` parameter is an array specifying the size of each
 * dimension in `in`. The `strides` parameter is an array specifying the stride
 * of each dimension in `in`.
 *
 * The function attempts to compress the dimensions of `in` based on the `shape`
 * and `strides` parameters. If all dimensions are fully compressed, meaning the
 * strides match the size of each dimension, the function performs a
 * copy-convert operation by invoking the `convert_copy` method of the `type`
 * attribute of `out`. Otherwise, the function iterates over the dimensions and
 * performs the following steps:
 * - Calculates the source offset based on the current dimension's index and
 * strides.
 * - Creates a temporary ScalarArray `tmp_in` representing a subarray of `in`
 * using the calculated source offset and the compression size, which is the
 * product of the remaining dimensions' sizes.
 * - Creates a temporary subarray `tmp_out` of `out` using the current offset
 * and the compression size.
 * - Invokes the `convert_copy` method of the `type` attribute of `out` to copy
 * and convert the data from `tmp_in` to `tmp_out`.
 *
 * @param[out] out The output ScalarArray where the data will be copied.
 * @param[in] in The input ScalarArray that provides the data to be copied.
 * @param[in] ndim The number of dimensions in the arrays.
 * @param[in] shape An array specifying the size of each dimension in `in`.
 * @param[in] strides An array specifying the stride of each dimension in `in`.
 *
 * @note This function assumes that `in` and `out` are valid ScalarArrays and
 * that `shape` and `strides` have at least `ndim` elements. The function
 * modifies `out` and does not modify `in`. This function is noexcept.
 */
void stride_copy(
        scalars::ScalarArray& out,
        const scalars::ScalarArray& in,
        const int32_t ndim,
        const idimn_t* shape,
        const idimn_t* strides
);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_STRIDED_COPY_H
