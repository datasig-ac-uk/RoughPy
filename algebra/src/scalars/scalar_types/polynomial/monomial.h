//
// Created by sam on 3/26/24.
//

#ifndef ROUGHPY_SCALAR_IMPLEMENTATIONS_MONOMIAL_H
#define ROUGHPY_SCALAR_IMPLEMENTATIONS_MONOMIAL_H

#include <roughpy/core/container/map.h>
#include <roughpy/core/hash.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <ostream>
#include <roughpy/platform/serialization.h>

namespace rpy {
namespace scalars {

namespace dtl {

/**
 * @class PackedInteger
 * @brief A class representing a packed integer
 *
 * This class allows for efficient storage and manipulation of integers
 * by packing a smaller integral type (Packed) into a larger integral type
 * (Base).
 *
 * PackedInteger supports various operations, such as comparison,
 * conversion to Packed and Base types, and serialization.
 *
 * @tparam Base The base integral type that the packed integer is stored in
 * @tparam Packed The packed integral type that is stored in the base type
 */
template <typename Base, typename Packed>
class PackedInteger
{
    Base m_data{};

    static_assert(
            is_integral_v<Base> && is_integral_v<Packed>,
            "both base and packed must be integral types"
    );

    static_assert(
            sizeof(Packed) < sizeof(Base),
            "size of packed type must be strictly less than base type"
    );

    static constexpr int32_t packed_bits = sizeof(Packed) * CHAR_BIT;
    static constexpr int32_t packed_offset
            = (sizeof(Base) - sizeof(Packed)) * CHAR_BIT;
    static constexpr Base packed_mask = (Base(1) << (packed_bits) -1)
            << packed_offset;

    static constexpr Base remaining_mask = Base(~packed_mask);

public:
    using packed_type = Packed;
    using integral_type = Base;

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

    constexpr explicit operator Base() const noexcept { return base(); }

    constexpr explicit operator Packed() const noexcept { return packed(); }

    bool operator==(const PackedInteger& other) const noexcept
    {
        return m_data == other.m_data;
    }

    bool operator!=(const PackedInteger& other) const noexcept
    {
        return m_data != other.m_data;
    }

    bool operator<(const PackedInteger& other) const noexcept
    {
        return m_data < other.m_data;
    }

    bool operator<=(const PackedInteger& other) const noexcept
    {
        return m_data <= other.m_data;
    }

    bool operator>(const PackedInteger& other) const noexcept
    {
        return m_data > other.m_data;
    }

    bool operator>=(const PackedInteger& other) const noexcept
    {
        return m_data >= other.m_data;
    }

    friend hash_t hash_value(const PackedInteger& arg) noexcept
    {
        return static_cast<hash_t>(arg.m_data);
    }

    RPY_SERIAL_SERIALIZE_FN();
};

template <typename Base, typename Packed>
template <typename Archive>
void PackedInteger<Base, Packed>::serialize(
        Archive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
)
{
    RPY_SERIAL_SERIALIZE_BARE(m_data);
}

template <typename Base, typename Packed>
inline std::ostream&
operator<<(std::ostream& os, const PackedInteger<Base, Packed>& letter)
{
    os << letter.packed();
    const auto id = letter.base();
    if (RPY_LIKELY(id != 0)) { os << id; }
    return os;
}

}// namespace dtl

using indeterminate_type = dtl::PackedInteger<deg_t, char>;

/**
 * @class Monomial
 * @brief A class representing a monomial
 */
class ROUGHPY_SCALARS_EXPORT Monomial
{

private:
    using map_type = containers::SmallFlatMap<indeterminate_type, deg_t, 1>;

    map_type m_data{};

public:
    using value_type = typename map_type::value_type;
    using reference = typename map_type::reference;
    using const_reference = typename map_type::const_reference;
    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;

    Monomial() = default;
    Monomial(const Monomial& other) = default;
    Monomial(Monomial&& other) noexcept = default;

    explicit Monomial(indeterminate_type letter, deg_t deg = 1)
        : m_data{
                  {letter, deg}
    }
    {}

    explicit Monomial(char symbol, deg_t id, deg_t degree = 1)
        : m_data{
                  {indeterminate_type(symbol, id), degree}
    }
    {}

    template <typename Rng>
    explicit Monomial(Rng&& range)
        : m_data(ranges::begin(range), ranges::end(range))
    {}

    Monomial& operator=(const Monomial& other) = default;
    Monomial& operator=(Monomial&& other) noexcept = default;

    /**
     * @brief Get the degree of the monomial.
     *
     * This method calculates and returns the degree of the monomial.
     * The degree is calculated by summing up the exponents of all the variables
     * in the monomial.
     *
     * @return The degree of the monomial as an integer of type deg_t.
     *
     * @see Monomial
     */
    RPY_NO_DISCARD deg_t degree() const noexcept
    {
        return ranges::accumulate(m_data | views::values, 0, std::plus<>());
    }

    /**
     * @brief Returns the type of the monomial
     *
     * They *type* of a monomial is the number of distinct indeterminates that
     * appear in the monomial.
     *
     * @return The type of the monomial as a deg_t type
     */
    RPY_NO_DISCARD deg_t type() const noexcept { return m_data.size(); }

    RPY_NO_DISCARD iterator begin() noexcept { return m_data.begin(); }
    RPY_NO_DISCARD iterator end() noexcept { return m_data.end(); }
    RPY_NO_DISCARD const_iterator begin() const noexcept
    {
        return m_data.begin();
    }
    RPY_NO_DISCARD const_iterator end() const noexcept { return m_data.end(); }

    RPY_NO_DISCARD deg_t operator[](const indeterminate_type& val
    ) const noexcept
    {
        auto it = m_data.find(val);
        if (it != m_data.end()) { return it->second; }
        return 0;
    }

    RPY_NO_DISCARD deg_t& operator[](const indeterminate_type& arg)
    {
        return m_data[arg];
    }

    Monomial& operator*=(const Monomial& rhs)
    {
        for (const auto& [key, value] : rhs.m_data) { m_data[key] += value; }
        return *this;
    }
    Monomial& operator*=(indeterminate_type rhs)
    {
        m_data[rhs] += 1;
        return *this;
    }

    RPY_SERIAL_SERIALIZE_FN();
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

inline std::ostream& operator<<(std::ostream& os, const Monomial& arg)
{
    bool first = true;
    for (const auto& item : arg) {
        if (first) {
            first = false;
        } else {
            os << ' ';
        }
        os << item.first;
        if (item.second > 1) { os << '^' << item.second; }
    }
    return os;
}

RPY_NO_DISCARD inline Monomial
operator*(const Monomial& lhs, const Monomial& rhs)
{
    Monomial result(lhs);
    result *= rhs;
    return result;
}

RPY_SERIAL_SERIALIZE_FN_IMPL(Monomial) { RPY_SERIAL_SERIALIZE_BARE(m_data); }

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALAR_IMPLEMENTATIONS_MONOMIAL_H
