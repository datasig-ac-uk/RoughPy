// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 23/05/23.
//

#include "rational_poly_scalar_type.h"
#include "scalar_pointer.h"
#include "scalar.h"

namespace rpy {
namespace scalars {
const ScalarType *RationalPolyScalarType::rational_type() const noexcept {
    return ScalarType::of<rational_scalar_type>();
}
const ScalarType *RationalPolyScalarType::host_type() const noexcept {
    return this;
}
Scalar RationalPolyScalarType::from(long long int numerator, long long int denominator) const {
    return Scalar(this, rational_poly_scalar(rational_scalar_type(numerator)/rational_scalar_type(denominator)));
}
ScalarPointer RationalPolyScalarType::allocate(std::size_t count) const {
    if (count == 1) {
        return ScalarPointer(this, new rational_poly_scalar(), flags::IsMutable | flags::OwnedPointer);
    } else {
        return ScalarPointer(this, new rational_poly_scalar[count](), flags::IsMutable | flags::OwnedPointer);
    }
}
void RationalPolyScalarType::free(ScalarPointer pointer, std::size_t count) const {
    if (!pointer.is_null()) {
        if (count == 1) {
            delete pointer.template raw_cast<rational_poly_scalar>();
        } else {
            delete[] pointer.template raw_cast<rational_poly_scalar>();
        }
    }
}
void RationalPolyScalarType::swap(ScalarPointer lhs, ScalarPointer rhs) const {
    RPY_CHECK(!(lhs.is_null() ^ rhs.is_null()));
    RPY_CHECK((lhs.type() == nullptr || rhs.type() == nullptr) || lhs.type() == rhs.type());
    RPY_CHECK(lhs.type() == this);
    RPY_CHECK(!lhs.is_const() && !rhs.is_const());

    std::swap(
        *lhs.raw_cast<rational_poly_scalar*>(),
        *rhs.raw_cast<rational_poly_scalar*>()
        );
}
void RationalPolyScalarType::convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count) const {
}
void RationalPolyScalarType::convert_copy(void *out, const void *in, std::size_t count, BasicScalarInfo info) const {
}
void RationalPolyScalarType::convert_copy(void *out, ScalarPointer in, std::size_t count) const {
}
void RationalPolyScalarType::convert_copy(ScalarPointer out, const void *in, std::size_t count, const string &id) const {
}
void RationalPolyScalarType::convert_fill(ScalarPointer out, ScalarPointer in, dimn_t count, const string &id) const {
    ScalarType::convert_fill(out, in, count, id);
}
Scalar RationalPolyScalarType::parse(string_view str) const {
    return ScalarType::parse(str);
}
Scalar RationalPolyScalarType::one() const {
    return ScalarType::one();
}
Scalar RationalPolyScalarType::mone() const {
    return ScalarType::mone();
}
Scalar RationalPolyScalarType::zero() const {
    return ScalarType::zero();
}
scalar_t RationalPolyScalarType::to_scalar_t(ScalarPointer arg) const {
    return 0;
}
void RationalPolyScalarType::assign(ScalarPointer target, long long int numerator, long long int denominator) const {
}
Scalar RationalPolyScalarType::copy(ScalarPointer source) const {
    return ScalarType::copy(source);
}
Scalar RationalPolyScalarType::uminus(ScalarPointer arg) const {
    return Scalar();
}
Scalar RationalPolyScalarType::add(ScalarPointer lhs, ScalarPointer rhs) const {
    return ScalarType::add(lhs, rhs);
}
Scalar RationalPolyScalarType::sub(ScalarPointer lhs, ScalarPointer rhs) const {
    return ScalarType::sub(lhs, rhs);
}
Scalar RationalPolyScalarType::mul(ScalarPointer lhs, ScalarPointer rhs) const {
    return ScalarType::mul(lhs, rhs);
}
Scalar RationalPolyScalarType::div(ScalarPointer lhs, ScalarPointer rhs) const {
    return ScalarType::div(lhs, rhs);
}
void RationalPolyScalarType::add_inplace(ScalarPointer lhs, ScalarPointer rhs) const {
}
void RationalPolyScalarType::sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const {
}
void RationalPolyScalarType::mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const {
}
void RationalPolyScalarType::div_inplace(ScalarPointer lhs, ScalarPointer rhs) const {
}
bool RationalPolyScalarType::is_zero(ScalarPointer arg) const {
    return ScalarType::is_zero(arg);
}
bool RationalPolyScalarType::are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept {
    return false;
}
void RationalPolyScalarType::print(ScalarPointer arg, std::ostream &os) const {
    ScalarType::print(arg, os);
}
std::unique_ptr<RandomGenerator> RationalPolyScalarType::get_rng(const string &bit_generator, Slice<uint64_t> seed) const {
    throw std::runtime_error("no rng for rational polynomial scalars");
}
std::unique_ptr<BlasInterface> RationalPolyScalarType::get_blas() const {
    throw std::runtime_error("no blas implementation for rational polynomial scalars");
}
std::vector<byte> RationalPolyScalarType::to_raw_bytes(const ScalarPointer &ptr, dimn_t count) const {
    return std::vector<byte>();
}
ScalarPointer RationalPolyScalarType::from_raw_bytes(Slice<byte> raw_bytes, dimn_t count) const {
    return ScalarPointer();
}
}// namespace scalars
}// namespace rpy
