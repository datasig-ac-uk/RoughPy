//
// Created by user on 25/07/22.
//

#ifndef LIBALGEBRA_LITE_INDEX_KEY_H
#define LIBALGEBRA_LITE_INDEX_KEY_H

#include "implementation_types.h"

#include <limits>
#include <ostream>
#include <unordered_map>

namespace lal {

namespace dtl {
struct index_key_access;
} // namespace dtl

template <int DegreeDigits=4, typename Int=dimn_t>
class index_key
{
    using limits = std::numeric_limits<Int>;
    static constexpr int degree_digits = DegreeDigits;

    static_assert(DegreeDigits < limits::digits, "DegreeDigits cannot exceed number of digits of Int");
    static constexpr int index_bits = limits::digits - DegreeDigits;
    static constexpr Int index_mask = (Int(1) << index_bits) - 1;
    static constexpr Int degree_mask = ~index_mask;

    Int m_data;

    friend struct dtl::index_key_access;

    explicit constexpr index_key(Int raw) : m_data(raw)
    {}

public:
    static constexpr deg_t max_degree = (deg_t(1) << degree_digits) - 1;

    using index_type = Int;

    explicit constexpr index_key() : m_data(0)
    {}

    explicit constexpr index_key(deg_t degree, index_type index)
        : m_data((Int(degree) << index_bits) + index)
    {}

    template <typename DegreeInt, typename IndexInt>
    explicit constexpr index_key(DegreeInt degree, IndexInt index)
        : m_data((Int(degree) << index_bits) + Int(index))
    {}

    constexpr Int index() const noexcept
    {
        return m_data & index_mask;
    }

    constexpr Int degree() const noexcept
    {
        return (m_data & degree_mask) >> index_bits;
    }

    constexpr bool operator==(const index_key& other) const noexcept
    { return m_data == other.m_data; }
    constexpr bool operator!=(const index_key& other) const noexcept
    { return m_data != other.m_data; }
    constexpr bool operator<(const index_key& other) const noexcept
    { return m_data < other.m_data; }
    constexpr bool operator<=(const index_key& other) const noexcept
    { return m_data <= other.m_data; }
    constexpr bool operator>(const index_key& other) const noexcept
    { return m_data > other.m_data; }
    constexpr bool operator>=(const index_key& other) const noexcept
    { return m_data >= other.m_data; }

    template <int OtherDegreeDigits, typename OtherType>
    constexpr bool operator==(const index_key<OtherDegreeDigits, OtherType>& other) const noexcept
    { return degree() == Int(other.degree()) && index() == Int(other.index()); }
    template <int OtherDegreeDigits, typename OtherType>
    constexpr bool operator!=(const index_key<OtherDegreeDigits, OtherType>& other) const noexcept
    { return degree() != Int(other.degree()) || index() != Int(other.index()); }
    template <int OtherDegreeDigits, typename OtherType>
    constexpr bool operator<(const index_key<OtherDegreeDigits, OtherType>& other) const noexcept
    { return degree() < Int(other.degree()) || (degree() == Int(other.degree()) && index() < Int(other.index())); }
    template<int OtherDegreeDigits, typename OtherType>
    constexpr bool operator<=(const index_key<OtherDegreeDigits, OtherType>& other) const noexcept
    { return degree()<Int(other.degree()) || (degree()==Int(other.degree()) && index()<=Int(other.index())); }
    template <int OtherDegreeDigits, typename OtherType>
    constexpr bool operator>(const index_key<OtherDegreeDigits, OtherType>& other) const noexcept
    { return degree()>Int(other.degree()) || (degree()==Int(other.degree()) && index()>Int(other.index())); }
    template<int OtherDegreeDigits, typename OtherType>
    constexpr bool operator>=(const index_key<OtherDegreeDigits, OtherType>& other) const noexcept
    { return degree()>Int(other.degree()) || (degree()==Int(other.degree()) && index()>=Int(other.index())); }

    index_key& operator++() noexcept
    {
        ++m_data;
        return *this;
    }

    friend std::size_t hash_value(const index_key& arg) noexcept
    {
        return arg.m_data;
    }

};

template <int DegreeDigits, typename Int>
std::ostream& operator<<(std::ostream& os, const index_key<DegreeDigits, Int>& arg) noexcept
{
    return os << "index_key(" << arg.degree() << ", " << arg.index() << ')';
}


namespace dtl {

struct index_key_access
{
    template <int DegreeDigits, typename Int>
    static constexpr Int raw(const index_key<DegreeDigits, Int>& arg) noexcept
    { return arg.m_data; }

    template <int DegreeDigits, typename Int>
    static constexpr index_key<DegreeDigits, Int> from_raw(Int raw) noexcept
    { return index_key<DegreeDigits, Int>(raw); }

};

} // namespace dtl


} // namespace lal

namespace std {

template <int DegreeDigits, typename Int>
struct hash<lal::index_key<DegreeDigits, Int>> {
    constexpr size_t operator()(const lal::index_key<DegreeDigits, Int>& arg) const noexcept
    { return size_t(lal::dtl::index_key_access::raw(arg)); }
};

}


#endif //LIBALGEBRA_LITE_INDEX_KEY_H
