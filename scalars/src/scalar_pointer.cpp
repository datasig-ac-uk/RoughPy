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
// Created by user on 26/02/23.
//

#include <roughpy/scalars/scalar_pointer.h>

#include <roughpy/core/alloc.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>

using namespace rpy;
using namespace rpy::scalars;

void* ScalarPointer::ptr()
{
    if (is_const()) {
        RPY_THROW(std::runtime_error,
                "attempting to convert const pointer to non-const pointer");
    }
    return const_cast<void*>(p_data);
}
Scalar ScalarPointer::deref() const noexcept
{
    return Scalar(*this, (m_flags & ~owning_flag) | constness_flag);
}
Scalar ScalarPointer::deref_mut()
{
    if (is_const()) {
        RPY_THROW(std::runtime_error,
                "attempting to dereference const pointer to non-const value");
    }
    return Scalar(*this, m_flags & ~owning_flag);
}
Scalar ScalarPointer::operator*() { return deref_mut(); }
Scalar ScalarPointer::operator*() const noexcept { return deref(); }
ScalarPointer
ScalarPointer::operator+(ScalarPointer::size_type index) const noexcept
{
    if (p_data == nullptr || p_type == nullptr) { return {}; }

    const auto* new_ptr
            = static_cast<const char*>(p_data) + index * p_type->itemsize();
    return {p_type, static_cast<const void*>(new_ptr), m_flags & ~owning_flag};
}
ScalarPointer&
ScalarPointer::operator+=(ScalarPointer::size_type index) noexcept
{
    if (p_data != nullptr && p_type != nullptr) {
        p_data = static_cast<const char*>(p_data) + index * p_type->itemsize();
    }
    return *this;
}
ScalarPointer& ScalarPointer::operator++() noexcept
{
    if (p_type != nullptr && p_data != nullptr) {
        p_data = static_cast<const char*>(p_data) + p_type->itemsize();
    }
    return *this;
}
const ScalarPointer ScalarPointer::operator++(int) noexcept
{
    ScalarPointer result(*this);
    this->operator++();
    return result;
}
Scalar ScalarPointer::operator[](ScalarPointer::size_type index) const noexcept
{
    return (*this + index).deref();
}
Scalar ScalarPointer::operator[](ScalarPointer::size_type index)
{
    return (*this + index).deref_mut();
}
ScalarPointer::difference_type
ScalarPointer::operator-(const ScalarPointer& right) const noexcept
{
    const ScalarType* type = p_type;
    if (type == nullptr) {
        if (right.p_type != nullptr) {
            type = right.p_type;
        } else {
            return 0;
        }
    }
    return static_cast<difference_type>(
                   static_cast<const char*>(p_data)
                   - static_cast<const char*>(right.p_data))
            / type->itemsize();
}

std::string rpy::scalars::ScalarPointer::get_type_id() const
{
    if (p_type != nullptr) { return p_type->id(); }
    RPY_CHECK(is_simple_integer());

    BasicScalarInfo info{
            is_signed_integer() ? ScalarTypeCode::Int : ScalarTypeCode::UInt,
            static_cast<uint8_t>(CHAR_BIT * simple_integer_bytes()), 1};

    return id_from_basic_info(info);
}
std::vector<byte> rpy::scalars::ScalarPointer::to_raw_bytes(dimn_t count) const
{
    if (p_type != nullptr) { return p_type->to_raw_bytes(*this, count); }

    RPY_CHECK(is_simple_integer());

    const auto n_bytes = count * simple_integer_bytes();
    std::vector<byte> result(n_bytes);
    std::memcpy(result.data(), p_data, n_bytes);
    return result;
}
void rpy::scalars::ScalarPointer::update_from_bytes(const std::string& type_id,
                                                    dimn_t count,
                                                    Slice<byte> raw)
{

    const auto* type = get_type(type_id);
    if (type != nullptr) {
        RPY_CHECK(count * type->itemsize() == raw.size());
        ScalarPointer::operator=(type->from_raw_bytes(raw, count));
        return;
    }

    // null type but no error says simple integer.
    const auto& info = get_scalar_info(type_id);
    RPY_CHECK(count * info.n_bytes == raw.size());

    p_data = aligned_alloc(info.alignment, raw.size());
    std::memcpy(const_cast<void*>(p_data), raw.begin(), raw.size());

    m_flags = flags::OwnedPointer;
    if (info.basic_info.code == ScalarTypeCode::Int) {
        m_flags |= flags::Signed;
    }
    auto order = static_cast<uint32_t>(count_bits(info.n_bytes));
    RPY_DBG_ASSERT((dimn_t(1) << order) == info.n_bytes);
    RPY_DBG_ASSERT(order <= 7);
    m_flags |= (order << integer_bits_offset);
    RPY_DBG_ASSERT(simple_integer_bytes() == info.n_bytes);
}
