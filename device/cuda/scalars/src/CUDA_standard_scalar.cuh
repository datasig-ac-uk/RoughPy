// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 22/04/23.
//

#ifndef ROUGHPY_DEVICE_CUDA_SCALARS_SRC_CUDA_STANDARD_SCALAR_CUH
#define ROUGHPY_DEVICE_CUDA_SCALARS_SRC_CUDA_STANDARD_SCALAR_CUH

#include <roughpy/device/device_core.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_pointer.h>
#include <roughpy/scalars/scalar_type.h>

namespace rpy {
namespace scalars {

template <typename S>
class CUDAScalarType : public ScalarType
{

    void copy_to_device(void* dp_dst, const void* hp_src, dimn_t count) const;
    void copy_from_device(void* hp_dst, const void* dp_src, dimn_t count) const;
    void
    copy_device_to_device(void* dp_dst, const void* dp_src, dimn_t count) const;

    inline bool check_types(const ScalarType* other) const
    {
        return other == this;
    }

    inline void handle_error(cudaError_t errcode) const {}

public:
    using ScalarType::ScalarType;

    ScalarPointer allocate(std::size_t count) const override;
    void free(ScalarPointer pointer, std::size_t count) const override;
    void swap(ScalarPointer lhs, ScalarPointer rhs) const override;
    void convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count)
            const override;
    void convert_copy(
            void* out, const void* in, std::size_t count, BasicScalarInfo info
    ) const override;
    void
    convert_copy(void* out, ScalarPointer in, std::size_t count) const override;
    void convert_copy(
            ScalarPointer out, const void* in, std::size_t count,
            const string& id
    ) const override;

    scalar_t to_scalar_t(ScalarPointer arg) const override;
    void
    assign(ScalarPointer target, long long int numerator,
           long long int denominator) const override;
    Scalar uminus(ScalarPointer arg) const override;
    void add_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void div_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    bool
    are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept override;
};

template <typename S>
void CUDAScalarType<S>::copy_to_device(
        void* dp_dst, const void* hp_src, dimn_t count
) const
{
    handle_error(cudaMemcpy(
            dp_dst, hp_src, count * sizeof(S), cudaMemcpyHostToDevice
    ));
}
template <typename S>
void CUDAScalarType<S>::copy_from_device(
        void* hp_dst, const void* dp_src, dimn_t count
) const
{
    handle_error(cudaMemcpy(
            hp_dst, dp_src, count * sizeof(S), cudaMemcpyDeviceToHost
    ));
}
template <typename S>
void CUDAScalarType<S>::copy_device_to_device(
        void* dp_dst, const void* dp_src, dimn_t count
) const
{
    handle_error(cudaMemcpy(
            dp_dst, dp_src, count * sizeof(S), cudaMemcpyDeviceToDevice
    ));
}

template <typename S>
ScalarPointer CUDAScalarType<S>::allocate(std::size_t count) const
{
    S* rptr;
    handle_error(cudaMalloc((void**) &rptr, count));

    return ScalarPointer(this, rptr, flags::OwnedPointer);
}
template <typename S>
void CUDAScalarType<S>::free(
        ScalarPointer pointer, std::size_t RPY_UNUSED_VAR count
) const
{
    handle_error(cudaFree(pointer.ptr()));
}
template <typename S>
void CUDAScalarType<S>::swap(ScalarPointer lhs, ScalarPointer rhs) const
{}
template <typename S>
void CUDAScalarType<S>::convert_copy(
        ScalarPointer dst, ScalarPointer src, dimn_t count
) const
{
    const auto* host_type = ScalarType::of<S>();
    RPY_DBG_ASSERT(!dst.is_null());
    RPY_DBG_ASSERT(!dst.is_const());
    RPY_DBG_ASSERT(!src.is_null());
    RPY_DBG_ASSERT(count > 0);

    if (dst.type() == this && src.type() == this) {
        handle_error(copy_device_to_device(dst.ptr(), src.cptr(), count));
    } else if (dst.type() == this && src.type() == host_type) {
        handle_error(copy_to_device(dst.ptr(), src.cptr(), count));
    } else if (dst.type() == host_type && src.type() == this) {
        handle_error(copy_from_device(dst.ptr(), src.cptr(), count));
    } else if (dst.type() == this) {
        std::vector<S> tmp(count);

        if (src.type() == nullptr) {
            if (!src.is_simple_integer()) {
                RPY_THROW(std::runtime_error, "unknown scalar type, cannot convert");
            }

            host_type->convert_copy({host_type, tmp.data()}, src, count);
        } else {
            src.type()->convert_copy({host_type, tmp.data()}, src, count);
        }

        handle_error(copy_to_device(dst.ptr(), tmp.data(), count));
    } else if (src.type() == this) {
        std::vector<S> tmp(count);
        handle_error(copy_from_device(tmp.data(), src.cptr(), count));
        dst.type()->convert_copy(dst, {host_type, tmp.data()}, count);
    } else {
        if (dst.type() == nullptr) {
            RPY_THROW(std::runtime_error, "destination type is null");
        }
        dst.type()->convert_copy(dst, src, count);
    }
}
template <typename S>
void CUDAScalarType<S>::convert_copy(
        void* out, const void* in, std::size_t count, BasicScalarInfo info
) const
{}
template <typename S>
void CUDAScalarType<S>::convert_copy(
        void* out, ScalarPointer in, std::size_t count
) const
{
    convert_copy({this, out}, in, count);
}
template <typename S>
void CUDAScalarType<S>::convert_copy(
        ScalarPointer out, const void* in, std::size_t count, const string& id
) const
{

    if (out.type() != this) {
        return out.type()->convert_copy(out, in, count, id);
    }
    const auto* host_type = ScalarType::of<S>();

    std::vector<S> tmp(count);
    host_type->convert_copy({host_type, tmp.data()}, in, count, id);

    handle_error(copy_to_device(out.ptr(), tmp.data(), count));
}
template <typename S>
scalar_t CUDAScalarType<S>::to_scalar_t(ScalarPointer arg) const
{
    S tmp;
    handle_error(copy_from_device(&tmp, arg.cptr(), 1));

    return scalar_t(tmp);
}
template <typename S>
void CUDAScalarType<S>::assign(
        ScalarPointer target, long long int numerator, long long int denominator
) const
{

    S tmp(numerator);
    tmp /= static_cast<S>(denominator);

    handle_error(copy_to_device(target.ptr()));
}
template <typename S>
Scalar CUDAScalarType<S>::uminus(ScalarPointer arg) const
{
    S* ptr;

    return Scalar();
}
template <typename S>
void CUDAScalarType<S>::add_inplace(ScalarPointer lhs, ScalarPointer rhs) const
{}
template <typename S>
void CUDAScalarType<S>::sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const
{}
template <typename S>
void CUDAScalarType<S>::mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const
{}
template <typename S>
void CUDAScalarType<S>::div_inplace(ScalarPointer lhs, ScalarPointer rhs) const
{}
template <typename S>
bool CUDAScalarType<S>::are_equal(ScalarPointer lhs, ScalarPointer rhs)
        const noexcept
{
    return false;
}

}// namespace scalars
}// namespace rpy
#endif// ROUGHPY_DEVICE_CUDA_SCALARS_SRC_CUDA_STANDARD_SCALAR_CUH
