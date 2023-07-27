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
// Created by user on 23/05/23.
//

#ifndef ROUGHPY_SCALARS_SRC_RATIONAL_POLY_SCALAR_TYPE_H
#define ROUGHPY_SCALARS_SRC_RATIONAL_POLY_SCALAR_TYPE_H

#include <roughpy/scalars/conversion.h>
#include <roughpy/scalars/scalar_type.h>

namespace rpy {
namespace scalars {

class RationalPolyScalarType : public ScalarType
{
    using scalar_type = rational_poly_scalar;

public:
    explicit RationalPolyScalarType()
        : ScalarType({
                string("RationalPoly"),
                string("RationalPoly"),
                sizeof(rational_poly_scalar),
                alignof(rational_poly_scalar),
                {ScalarTypeCode::OpaqueHandle, 0, 0},
                {ScalarDeviceType::CPU, 0},
    })
    {}

    const ScalarType* rational_type() const noexcept override;
    const ScalarType* host_type() const noexcept override;
    Scalar
    from(long long int numerator, long long int denominator) const override;
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
    void convert_fill(
            ScalarPointer out, ScalarPointer in, dimn_t count, const string& id
    ) const override;
    Scalar parse(string_view str) const override;
    Scalar one() const override;
    Scalar mone() const override;
    Scalar zero() const override;
    scalar_t to_scalar_t(ScalarPointer arg) const override;
    void
    assign(ScalarPointer target, long long int numerator,
           long long int denominator) const override;
    Scalar copy(ScalarPointer source) const override;
    Scalar uminus(ScalarPointer arg) const override;
    Scalar add(ScalarPointer lhs, ScalarPointer rhs) const override;
    Scalar sub(ScalarPointer lhs, ScalarPointer rhs) const override;
    Scalar mul(ScalarPointer lhs, ScalarPointer rhs) const override;
    Scalar div(ScalarPointer lhs, ScalarPointer rhs) const override;
    void add_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void div_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;

    void add_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const override;
    void sub_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const override;
    void mul_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const override;
    void div_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const override;

    bool is_zero(ScalarPointer arg) const override;
    bool
    are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept override;
    void print(ScalarPointer arg, std::ostream& os) const override;
    std::unique_ptr<RandomGenerator>
    get_rng(const string& bit_generator, Slice<uint64_t> seed) const override;
    std::unique_ptr<BlasInterface> get_blas() const override;
    std::vector<byte>
    to_raw_bytes(const ScalarPointer& ptr, dimn_t count) const override;
    ScalarPointer
    from_raw_bytes(Slice<byte> raw_bytes, dimn_t count) const override;
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_RATIONAL_POLY_SCALAR_TYPE_H
