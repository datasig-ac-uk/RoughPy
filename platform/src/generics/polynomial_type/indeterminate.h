//
// Created by sam on 26/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_INDETERMINATE_H
#define ROUGHPY_GENERICS_INTERNAL_INDETERMINATE_H

#include <cstring>
#include <iosfwd>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/hash.h"
#include "roughpy/core/helpers.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"


namespace rpy::generics {

class Indeterminate
{
    using base_type = uint64_t;
    base_type m_data;

public:
    static constexpr size_t data_size = sizeof(base_type);
    static constexpr size_t num_chars = 1;

    static_assert(
            num_chars >= 1,
            "sanity check failed: at least one prefix character"
    );
    static_assert(
            data_size > num_chars,
            "sanity check failed, base type be large than num_chars bytes"
    );

    static constexpr int char_shift
            = static_cast<int>(data_size - num_chars) * 8;

    static constexpr base_type integer_mask
            = (static_cast<base_type>(1) << char_shift) - 1;

    static constexpr base_type default_value =
        static_cast<base_type>('x') << char_shift;


    constexpr Indeterminate() noexcept : m_data(default_value) {}

    explicit Indeterminate(base_type integer) noexcept
        : m_data(default_value | (integer & integer_mask))
    {
        RPY_DBG_ASSERT_LE(integer, integer_mask);
    }

    explicit Indeterminate(char prefix, base_type integer) noexcept
    {
        RPY_DBG_ASSERT_LE(integer, integer_mask);

        m_data = static_cast<base_type>(prefix) << char_shift;
        m_data |= (integer & integer_mask);
    }

    template <size_t N, typename = enable_if_t<(N <= num_chars)>>
    Indeterminate(char prefix[N], base_type base) noexcept
    {
        RPY_DBG_ASSERT_LE(base, integer_mask);

        m_data = 0;
        std::memcpy(&m_data, prefix, N);
        m_data <<= char_shift;
        m_data |= (base & integer_mask);
    }

    string prefix() const
    {
        // This will almost surely never throw because the number of characters
        // should never exceed the size of the string internal storage
        string result;

        char tmp_array[num_chars];
        base_type tmp = m_data >> char_shift;
        std::memcpy(tmp_array, &tmp, num_chars);

        result.assign(tmp_array, num_chars);

        return result;
    }

    base_type index() const noexcept { return m_data & integer_mask; }

    friend constexpr bool
    operator==(const Indeterminate& lhs, const Indeterminate& rhs) noexcept
    {
        return lhs.m_data == rhs.m_data;
    }

    friend constexpr bool
    operator!=(const Indeterminate& lhs, const Indeterminate& rhs) noexcept
    {
        return lhs.m_data != rhs.m_data;
    }

    friend constexpr bool
    operator <(const Indeterminate& lhs, const Indeterminate& rhs) noexcept
    {
        return lhs.m_data < rhs.m_data;
    }

    friend constexpr bool
    operator <=(const Indeterminate& lhs, const Indeterminate& rhs) noexcept
    {
        return lhs.m_data <= rhs.m_data;
    }

    friend constexpr bool
    operator >(const Indeterminate& lhs, const Indeterminate& rhs) noexcept
    {
        return lhs.m_data > rhs.m_data;
    }

    friend constexpr bool
    operator >=(const Indeterminate& lhs, const Indeterminate& rhs) noexcept
    {
        return lhs.m_data >= rhs.m_data;
    }

    friend hash_t hash_value(const Indeterminate& value) noexcept {
        Hash<base_type> hasher;
        return hasher(value.m_data);
    }

};

std::ostream& operator<<(std::ostream& os, const Indeterminate& value);




};// namespace rpy::generics

#endif// ROUGHPY_GENERICS_INTERNAL_INDETERMINATE_H
