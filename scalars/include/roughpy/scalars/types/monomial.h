//
// Created by sam on 3/26/24.
//

#ifndef MONOMIAL_H
#define MONOMIAL_H

#include <roughpy/scalars/scalars_fwd.h>

#include <roughpy/core/container/map.h>
#include <roughpy/core/hash.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <ostream>

namespace rpy {
namespace scalars {

namespace dtl {

template <typename Base, typename Packed>
class PackedInteger
{
    Base m_data;

    static_assert(
            is_integral<Base>::value && is_integral<Packed>::value,
            "both base and packed must be integral types"
    );

    static_assert(
            sizeof(Packed) < sizeof(Base),
            "size of packed type must be strictly less than base type"
    );

    static constexpr int32_t packed_offset
            = (sizeof(Base) - sizeof(Packed)) * CHAR_BIT;
    static constexpr Base packed_mask
            = (Base(1) << (sizeof(Packed) * CHAR_BIT + 1 - 1)) << packed_offset;

    static constexpr Base remaining_mask = Base(~packed_mask);

public:
    PackedInteger(Packed packed, Base base)
        : m_data((static_cast<Base>(packed) << packed_offset) | (base))
    {
        RPY_CHECK((base & remaining_mask) == base);
    }

    explicit PackedInteger(Packed packed)
        : m_data(static_cast<Base>(packed) << packed_offset)
    {}

    constexpr Base base() const noexcept { return m_data & remaining_mask; }
    constexpr Packed packed() const noexcept
    {
        return static_cast<Packed>(m_data >> packed_offset);
    }
};

}// namespace dtl

class ROUGHPY_SCALARS_EXPORT Monomial
{
public:
    using letter_type = dtl::PackedInteger<deg_t, char>;

private:
    using map_type = containers::SmallFlatMap<letter_type, deg_t, 1>;

    map_type m_data;

public:
    using value_type = typename map_type::value_type;
    using reference = typename map_type::reference;
    using const_reference = typename map_type::const_reference;
    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;

    explicit Monomial(letter_type letter, deg_t deg = 1)
        : m_data{
                  {letter, deg}
    }
    {}

    explicit Monomial(char symbol, deg_t id, deg_t degree = 1)
        : m_data{
                  {letter_type(symbol, id), degree}
    }
    {}

    template <typename Rng>
    explicit Monomial(Rng&& range)
        : m_data(ranges::begin(range), ranges::end(range))
    {}

    RPY_NO_DISCARD deg_t degree() const noexcept;

    RPY_NO_DISCARD deg_t type() const noexcept { return m_data.size(); }

    RPY_NO_DISCARD iterator begin() noexcept { return m_data.begin(); }
    RPY_NO_DISCARD iterator end() noexcept { return m_data.end(); }
    RPY_NO_DISCARD const_iterator begin() const noexcept
    {
        return m_data.begin();
    }
    RPY_NO_DISCARD const_iterator end() const noexcept { return m_data.end(); }

    Monomial& operator*=(const Monomial& rhs);
    Monomial& operator*=(letter_type rhs);
};

RPY_NO_DISCARD inline bool
operator==(const Monomial& lhs, const Monomial& rhs) noexcept
{
    return lhs.degree() == rhs.degree() && ranges::equal(lhs, rhs);
}

RPY_NO_DISCARD inline bool
operator!=(const Monomial& lhs, const Monomial& rhs) noexcept
{
    return lhs.degree() != rhs.degree() || !ranges::equal(lhs, rhs);
}

RPY_NO_DISCARD inline bool
operator<(const Monomial& lhs, const Monomial& rhs) noexcept
{
    const auto ldegree = lhs.degree();
    const auto rdegree = rhs.degree();
    return (ldegree < rdegree)
            || (ldegree == rdegree && ranges::lexicographical_compare(lhs, rhs)
            );
}

RPY_NO_DISCARD inline bool
operator<=(const Monomial& lhs, const Monomial& rhs) noexcept
{
    const auto ldegree = lhs.degree();
    const auto rdegree = rhs.degree();
    return (ldegree < rdegree)
            || (ldegree == rdegree
                && ranges::lexicographical_compare(
                        lhs,
                        rhs,
                        std::less_equal<>()
                ));
}

RPY_NO_DISCARD inline bool
operator>(const Monomial& lhs, const Monomial& rhs) noexcept
{
    return rhs < lhs;
}
RPY_NO_DISCARD inline bool
operator>=(const Monomial& lhs, const Monomial& rhs) noexcept
{
    return rhs <= lhs;
}

RPY_NO_DISCARD inline hash_t hash_value(const Monomial& arg) noexcept
{
    hash_t result = 0;

    ranges::for_each(arg, [&result](const typename Monomial::value_type& val) {
        hash_combine(result, val);
    });

    return result;
}

ROUGHPY_SCALARS_EXPORT std::ostream&
operator<<(std::ostream& os, const Monomial& arg);

RPY_NO_DISCARD inline Monomial
operator*(const Monomial& lhs, const Monomial& rhs)
{
    Monomial result(lhs);
    result *= rhs;
    return result;
}

}// namespace scalars
}// namespace rpy

#endif// MONOMIAL_H
