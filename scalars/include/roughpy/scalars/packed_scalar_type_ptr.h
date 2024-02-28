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

#ifndef ROUGHPY_SCALARS_PACKED_SCALAR_TYPE_PTR_H_
#define ROUGHPY_SCALARS_PACKED_SCALAR_TYPE_PTR_H_

#include "scalars_fwd.h"

#include <roughpy/core/helpers.h>
#include <roughpy/core/traits.h>
// #include "scalar_type.h"

#include <limits>

namespace rpy {
namespace scalars {

template <typename OptionsEnumerator>
class PackedScalarTypePointer
{
    using integral_type = uint64_t;
    using enumeration_type = OptionsEnumerator;

    static_assert(sizeof(integral_type) >= sizeof(ScalarType*), "");
    static_assert(sizeof(uintptr_t) == sizeof(ScalarType*), "");

    using type_info_t = devices::TypeInfo;

    integral_type m_data = 0;

    static constexpr integral_type PointerBits = CHAR_BIT * sizeof(void*);
    static constexpr integral_type NonPointerBits
            = ConstLog2<min_scalar_type_alignment>::value;
    static_assert(
            NonPointerBits > 1,
            "at least one bit of space is required "
            "for a packed scalar type pointer"
    );

    static constexpr integral_type MaxEnumBits = NonPointerBits - 1;
    static constexpr integral_type EnumMask
            = (integral_type(1) << MaxEnumBits) - 1;
    static constexpr integral_type ModeMask = integral_type(1) << MaxEnumBits;

    // Pointer might be smaller than uint64_t, so mask this properly
    static constexpr integral_type PointerMask
            = std::numeric_limits<uintptr_t>::max() ^ (ModeMask | EnumMask);

    // If this isn't a pointer, then hide the TypeInfo struct in bits [36...4]
    static constexpr integral_type SizeOfTypeInfo
            = CHAR_BIT * sizeof(type_info_t);
    static_assert(
            SizeOfTypeInfo + NonPointerBits <= CHAR_BIT * sizeof(integral_type),
            ""
    );
    static constexpr integral_type TypeInfoMask
            = ((integral_type(1) << SizeOfTypeInfo) - 1) << NonPointerBits;

public:
    PackedScalarTypePointer() = default;

    constexpr
    PackedScalarTypePointer(const ScalarType* dtype, enumeration_type value)
        : m_data(bit_cast<integral_type>(dtype)
                 | static_cast<integral_type>(value))
    {}

    constexpr PackedScalarTypePointer(type_info_t info, enumeration_type ty)
        : m_data((static_cast<integral_type>(bit_cast<uint32_t>(info))
                  << MaxEnumBits)
                 | ModeMask | (static_cast<integral_type>(ty) & EnumMask))
    {}

    constexpr bool is_pointer() const noexcept
    {
        return (m_data & ModeMask) == 0;
    }

    constexpr bool is_null() const noexcept { return m_data == 0; }

    constexpr const ScalarType* get_pointer() const noexcept
    {
        return bit_cast<const ScalarType*>(
                static_cast<uintptr_t>(m_data & PointerMask)
        );
    }

    constexpr enumeration_type get_enumeration() const noexcept
    {
        return static_cast<enumeration_type>(m_data & EnumMask);
    }

    constexpr type_info_t get_type_info() const noexcept
    {
        return bit_cast<type_info_t>(
                static_cast<uint32_t>((m_data & TypeInfoMask) >> MaxEnumBits)
        );
    }

    constexpr const ScalarType& operator*() const noexcept
    {

        return *get_pointer();
    }

    constexpr const ScalarType* operator->() const noexcept
    {
        return get_pointer();
    }

    constexpr operator const ScalarType*() const noexcept
    {
        return get_pointer();
    }

    constexpr operator enumeration_type() const noexcept
    {
        return get_enumeration();
    }

    constexpr operator type_info_t() const noexcept { return get_type_info(); }

    void update_enumeration(enumeration_type enum_val) noexcept
    {
        m_data = (m_data & (PointerMask | ModeMask))
                | (static_cast<integral_type>(enum_val) & EnumMask);
    }

    friend constexpr bool operator==(const PackedScalarTypePointer& lhs, const PackedScalarTypePointer& rhs) noexcept
    {
        return lhs.m_data == rhs.m_data;
    }

};

template <typename OptionsEnumeration>
inline devices::TypeInfo type_info_from(const PackedScalarTypePointer<OptionsEnumeration>& arg) noexcept
{
    if (arg.is_pointer()) {
        return arg->type_info();
    }
    return arg.get_type_info();
}


}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_PACKED_SCALAR_TYPE_PTR_H_
