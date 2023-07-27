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

#ifndef ROUGHPY_SCALARS_SCALAR_POINTER_H_
#define ROUGHPY_SCALARS_SCALAR_POINTER_H_

#include "scalars_fwd.h"

#include <roughpy/core/helpers.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace scalars {

namespace flags {

enum Constness : uint32_t
{
    IsMutable = 0,
    IsConst = 1,
};

enum PointerType : uint32_t
{
    BorrowedPointer = 0,
    OwnedPointer = 2,
    InterfacePointer = 4,
    SimpleInteger = 8
};

enum IntegerSign : uint32_t
{
    Unsigned = 0,
    Signed = 1 << 7
};

enum IntegerType : uint32_t
{
    UnsignedInteger8 = Unsigned | 0 << 4 | SimpleInteger, // (2**0) bytes
    UnsignedInteger16 = Unsigned | 1 << 4 | SimpleInteger,// (2**1) bytes
    UnsignedInteger32 = Unsigned | 2 << 4 | SimpleInteger,// (2**2) bytes
    UnsignedInteger64 = Unsigned | 3 << 4 | SimpleInteger,// (2**3) bytes
    UnsignedSize = Unsigned | 4 << 4 | SimpleInteger,     // Special
    //  UnsignedUnused      = Unsigned | 5 << 4 | SimpleInteger,
    //  UnsignedUnused      = Unsigned | 6 << 4 | SimpleInteger,
    //  UnsignedUnused      = Unsigned | 7 << 4 | SimpleInteger,
    SignedInteger8 = Signed | 0 << 4 | SimpleInteger, // (2**0) bytes
    SignedInteger16 = Signed | 1 << 4 | SimpleInteger,// (2**1) bytes
    SignedInteger32 = Signed | 2 << 4 | SimpleInteger,// (2**2) bytes
    SignedInteger64 = Signed | 3 << 4 | SimpleInteger,// (2**3) bytes
    SignedSize = Signed | 4 << 4 | SimpleInteger      // Special
    //  SignedUnused        =   Signed | 5 << 4 | SimpleInteger,
    //  SignedUnused        =   Signed | 6 << 4 | SimpleInteger,
    //  SignedUnused        =   Signed | 7 << 4 | SimpleInteger
};

}// namespace flags

class ScalarPointer
{
    using Constness = flags::Constness;

protected:
    static constexpr uint32_t constness_flag = 1;
    static constexpr uint32_t owning_flag = 1 << 1;
    static constexpr uint32_t interface_flag = 1 << 2;
    static constexpr uint32_t integer_type_flag = 1 << 3;
    static constexpr uint32_t integer_bits_0 = 1 << 4;
    static constexpr uint32_t integer_bits_1 = 1 << 5;
    static constexpr uint32_t integer_bits_2 = 1 << 6;
    static constexpr uint32_t signed_flag = 1 << 7;

    static constexpr uint32_t pointer_flags = constness_flag | interface_flag
            | owning_flag | integer_type_flag | integer_bits_0 | integer_bits_1
            | integer_bits_2 | signed_flag;

    static constexpr uint32_t integer_bits_offset = 4;
    static constexpr uint32_t integer_bits_mask
            = integer_bits_0 | integer_bits_1 | integer_bits_2;

    static constexpr uint32_t subtype_flag_offset = 8;
    static constexpr uint32_t subtype_flag_mask = 0xF << subtype_flag_offset;

public:
protected:
    const ScalarType* p_type = nullptr;
    const void* p_data = nullptr;
    uint32_t m_flags = 0;

    //    Constness m_constness = IsMutable;

public:
    ScalarPointer(const ScalarType* type, const void* data, Constness constness)
        : p_type(type), p_data(data),
          m_flags(flags::BorrowedPointer | constness)
    {}

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    ScalarPointer() = default;

    ScalarPointer(const ScalarType* type, void* data, uint32_t flag)
        : p_type(type), p_data(data), m_flags(flag)
    {}

    ScalarPointer(const ScalarType* type, const void* data, uint32_t flag)
        : p_type(type), p_data(data), m_flags(flag)
    {}

    explicit ScalarPointer(const ScalarType* type) : p_type(type) {}

    ScalarPointer(const ScalarType* type, void* ptr)
        : p_type(type), p_data(ptr),
          m_flags(flags::BorrowedPointer | flags::IsMutable)
    {}
    ScalarPointer(const ScalarType* type, const void* ptr)
        : p_type(type), p_data(ptr),
          m_flags(flags::BorrowedPointer | flags::IsConst)
    {}

    explicit ScalarPointer(uint8_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::UnsignedInteger8 | flags::BorrowedPointer
                  | flags::IsMutable)
    {}

    explicit ScalarPointer(const uint8_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::UnsignedInteger8 | flags::BorrowedPointer
                  | flags::IsConst)
    {}

    explicit ScalarPointer(int8_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::SignedInteger8 | flags::BorrowedPointer
                  | flags::IsMutable)
    {}

    explicit ScalarPointer(const int8_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::SignedInteger8 | flags::BorrowedPointer
                  | flags::IsConst)
    {}

    explicit ScalarPointer(uint16_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::UnsignedInteger16 | flags::BorrowedPointer
                  | flags::IsMutable)
    {}

    explicit ScalarPointer(const uint16_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::UnsignedInteger16 | flags::BorrowedPointer
                  | flags::IsConst)
    {}

    explicit ScalarPointer(int16_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::SignedInteger16 | flags::BorrowedPointer
                  | flags::IsMutable)
    {}

    explicit ScalarPointer(const int16_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::SignedInteger16 | flags::BorrowedPointer
                  | flags::IsConst)
    {}

    explicit ScalarPointer(uint32_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::UnsignedInteger32 | flags::BorrowedPointer
                  | flags::IsMutable)
    {}

    explicit ScalarPointer(const uint32_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::UnsignedInteger32 | flags::BorrowedPointer
                  | flags::IsConst)
    {}

    explicit ScalarPointer(int32_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::SignedInteger32 | flags::BorrowedPointer
                  | flags::IsMutable)
    {}

    explicit ScalarPointer(const int32_t* ptr)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::SignedInteger32 | flags::BorrowedPointer
                  | flags::IsConst)
    {}

    explicit ScalarPointer(size_t* ptr, unsigned_size_type_marker)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::UnsignedSize | flags::BorrowedPointer
                  | flags::IsMutable)
    {}

    explicit ScalarPointer(const size_t* ptr, unsigned_size_type_marker)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::UnsignedSize | flags::BorrowedPointer | flags::IsConst)
    {}

    explicit ScalarPointer(ptrdiff_t* ptr, signed_size_type_marker)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::SignedSize | flags::BorrowedPointer | flags::IsMutable)
    {}

    explicit ScalarPointer(const ptrdiff_t* ptr, signed_size_type_marker)
        : p_type(nullptr), p_data(ptr),
          m_flags(flags::SignedSize | flags::BorrowedPointer | flags::IsConst)
    {}


    /**
     * @brief Get the raw flags from this pointer
     * @return uint32_t holding flags data
     */
    RPY_NO_DISCARD constexpr uint32_t flags() const noexcept { return m_flags; }

    /**
     * @brief Get whether the pointer is const or not
     * @return bool, true if pointer is const
     */
    RPY_NO_DISCARD bool is_const() const noexcept
    {
        return (m_flags & constness_flag) != 0;
    }

    /**
     * @brief Get whether the pointer is the null pointer
     * @return bool, true if the underlying raw pointer is null
     */
    RPY_NO_DISCARD bool is_null() const noexcept { return p_data == nullptr; }

    /**
     * @brief Get a pointer to the type of this scalar
     * @return pointer to type object
     */
    RPY_NO_DISCARD const ScalarType* type() const noexcept { return p_type; }

    RPY_NO_DISCARD constexpr bool is_simple_integer() const noexcept
    {
        return (m_flags & flags::SimpleInteger) != 0;
    }

    RPY_NO_DISCARD constexpr uint32_t simple_integer_bytes() const noexcept
    {
        return 1U << ((m_flags & integer_bits_mask) >> integer_bits_offset);
    }

    RPY_NO_DISCARD constexpr bool is_signed_integer() const noexcept
    {
        return (m_flags & signed_flag) != 0;
    }

    RPY_NO_DISCARD constexpr flags::IntegerType
    simple_integer_config() const noexcept
    {
        return static_cast<flags::IntegerType>(
                m_flags & (flags::SimpleInteger | signed_flag | integer_bits_mask)
        );
    }

    /**
     * @brief Get the raw pointer contained held
     * @return const raw pointer to underlying data
     */
    RPY_NO_DISCARD const void* ptr() const noexcept { return p_data; }

    /**
     * @brief Get the raw pointer contained held
     * @return const raw pointer to underlying data
     */
    RPY_NO_DISCARD const void* cptr() const noexcept { return p_data; }

    /**
     * @brief Get the mutable raw pointer held
     * @return mutable raw pointer to underlying data
     */
    RPY_NO_DISCARD void* ptr();

    /**
     * @brief Cast the pointer to a const raw type
     * @tparam T Type to cast to
     * @return pointer to underlying type of T
     */
    template <typename T>
    RPY_NO_DISCARD ensure_pointer<T> raw_cast() const noexcept
    {
        return static_cast<ensure_pointer<T>>(p_data);
    }

    /**
     * @brief Cast the pointer to a raw type
     * @tparam T Type to cast to
     * @return pointer to underlying type of T
     */
    template <
            typename T,
            typename
            = std::enable_if_t<!std::is_const<std::remove_pointer_t<T>>::value>>
    RPY_NO_DISCARD ensure_pointer<T> raw_cast()
    {
        if (is_const()) {
            RPY_THROW(
                    std::runtime_error,
                    "cannot cast const pointer to non-const pointer"
            );
        }
        return static_cast<ensure_pointer<T>>(const_cast<void*>(p_data));
    }

    /**
     * @brief Dereference to get a scalar value
     * @return new Scalar object referencing the pointed to data
     */
    RPY_NO_DISCARD Scalar deref() const noexcept;

    /**
     * @brief Dereference to a scalar type mutably
     * @return new Scalar object mutably referencing te pointed to data
     */
    RPY_NO_DISCARD Scalar deref_mut();

    // Pointer-like operations

    RPY_NO_DISCARD constexpr operator bool() const noexcept
    {
        return p_data != nullptr;
    }

    RPY_NO_DISCARD Scalar operator*();
    RPY_NO_DISCARD Scalar operator*() const noexcept;

    RPY_NO_DISCARD ScalarPointer operator+(size_type index) const noexcept;
    ScalarPointer& operator+=(size_type index) noexcept;

    ScalarPointer& operator++() noexcept;
    RPY_NO_DISCARD const ScalarPointer operator++(int) noexcept;

    RPY_NO_DISCARD Scalar operator[](size_type index) const noexcept;
    RPY_NO_DISCARD Scalar operator[](size_type index);

    RPY_NO_DISCARD difference_type operator-(const ScalarPointer& right
    ) const noexcept;

protected:
    RPY_NO_DISCARD constexpr bool is_owning() const noexcept
    {
        return (m_flags & owning_flag) != 0;
    }

    RPY_NO_DISCARD constexpr bool is_interface() const noexcept
    {
        return (m_flags & interface_flag) != 0;
    }

protected:
    RPY_NO_DISCARD std::string get_type_id() const;

    RPY_NO_DISCARD std::vector<byte> to_raw_bytes(dimn_t count) const;

    void update_from_bytes(
            const std::string& type_id, dimn_t count, Slice<byte> raw
    );
};

RPY_NO_DISCARD inline bool
operator==(const ScalarPointer& left, const ScalarPointer& right)
{
    return left.type() == right.type() && left.ptr() == right.ptr();
}

RPY_NO_DISCARD inline bool
operator!=(const ScalarPointer& left, const ScalarPointer& right)
{
    return left.type() != right.type() || left.ptr() != right.ptr();
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_POINTER_H_
