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
// Created by user on 01/11/23.
//
#ifndef ROUGHPY_SCALARS_SRC_SCALAR_ARITHMETIC_H_
#define ROUGHPY_SCALARS_SRC_SCALAR_ARITHMETIC_H_

#include "packed_type.h"
#include <roughpy/core/macros.h>
#include <roughpy/scalars/scalar.h>

namespace rpy {
namespace scalars {
namespace dtl {

void RPY_LOCAL scalar_inplace_add(void*, PackedType, const void*, PackedType);
void RPY_LOCAL scalar_inplace_sub(void*, PackedType, const void*, PackedType);
void RPY_LOCAL scalar_inplace_mul(void*, PackedType, const void*, PackedType);
void RPY_LOCAL scalar_inplace_div(void*, PackedType, const void*, PackedType);

template <typename D, typename S>
RPY_LOCAL void
scalar_inplace_add(void* dst, D dst_type, const void* src, S src_type)
{
    scalar_inplace_add(dst, pack_type(dst_type), src, pack_type(src_type));
}

template <typename D, typename S>
RPY_LOCAL void
scalar_inplace_sub(void* dst, D dst_type, const void* src, S src_type)
{
    scalar_inplace_sub(dst, pack_type(dst_type), src, pack_type(src_type));
}

template <typename D, typename S>
RPY_LOCAL void
scalar_inplace_mul(void* dst, D dst_type, const void* src, S src_type)
{
    scalar_inplace_mul(dst, pack_type(dst_type), src, pack_type(src_type));
}

template <typename D, typename S>
RPY_LOCAL void
scalar_inplace_div(void* dst, D dst_type, const void* src, S src_type)
{
    scalar_inplace_div(dst, pack_type(dst_type), src, pack_type(src_type));
}

}// namespace dtl
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_SCALAR_ARITHMETIC_H_
