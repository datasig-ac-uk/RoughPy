//
// Created by sam on 26/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_MONOMIAL_H
#define ROUGHPY_GENERICS_INTERNAL_MONOMIAL_H

#include <functional>
#include <iosfwd>

#include <boost/container/flat_map.hpp>
#include <boost/container/small_vector.hpp>

#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/hash.h"
#include "roughpy/core/types.h"


#include "indeterminate.h"

namespace rpy {
namespace generics {


class Monomial
{
    using map_storage_type = boost::container::
            small_vector<pair<Indeterminate, deg_t>, 1>;
    using map_type = boost::container::
            flat_map<Indeterminate, deg_t, std::less<>, map_storage_type>;

    map_type m_data;

public:

    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;

    Monomial() = default;

    explicit Monomial(Indeterminate indeterminate, deg_t degree=1) noexcept
    {
        m_data.emplace(std::move(indeterminate), std::move(degree));
    }

    explicit Monomial(char prefix, size_t index, deg_t degree=1) noexcept
    {
        m_data.emplace(Indeterminate(prefix, index), std::move(degree));
    }

    template <typename InputIt>
    explicit Monomial(InputIt begin, InputIt end)
        : m_data(begin, end)
    {}


    RPY_NO_DISCARD
    deg_t degree() const noexcept;
    RPY_NO_DISCARD
    deg_t type() const noexcept { return m_data.size(); }

    void clear() noexcept { m_data.clear(); }
    RPY_NO_DISCARD bool empty() const noexcept { return m_data.empty(); }

    RPY_NO_DISCARD
    iterator begin() noexcept { return m_data.begin(); }
    RPY_NO_DISCARD
    iterator end() noexcept { return m_data.end(); }
    RPY_NO_DISCARD
    const_iterator begin() const noexcept { return m_data.begin(); }
    RPY_NO_DISCARD
    const_iterator end() const noexcept { return m_data.end(); }

    template <typename... Args>
    auto emplace(Args&&... args) -> decltype(m_data.emplace(std::forward<Args>(args)...))
    {
        return m_data.emplace(std::forward<Args>(args)...);
    }

    RPY_NO_DISCARD
    deg_t operator[](Indeterminate indeterminate) const noexcept;
    RPY_NO_DISCARD
    deg_t& operator[](Indeterminate indeterminate) noexcept {
        return m_data[indeterminate];
    };

    Monomial& operator*=(const Monomial& rhs);


    friend hash_t hash_value(const Monomial& value);


};


RPY_NO_DISCARD
hash_t hash_value(const Monomial& value);

RPY_NO_DISCARD
Monomial operator*(const Monomial& lhs, const Monomial& rhs);
std::ostream &operator<<(std::ostream &os, const Monomial& value);

RPY_NO_DISCARD
bool operator==(const Monomial& lhs, const Monomial& rhs) noexcept;

RPY_NO_DISCARD
inline bool operator!=(const Monomial& lhs, const Monomial& rhs) noexcept
{
    return !(lhs == rhs);
}

RPY_NO_DISCARD
bool operator<(const Monomial& lhs, const Monomial& rhs) noexcept;
RPY_NO_DISCARD
bool operator<=(const Monomial& lhs, const Monomial& rhs) noexcept;

RPY_NO_DISCARD
inline bool operator>(const Monomial& lhs, const Monomial& rhs) noexcept
{
    return !(lhs <= rhs);
}

RPY_NO_DISCARD
inline bool operator>=(const Monomial& lhs, const Monomial& rhs) noexcept
{
    return !(lhs < rhs);
}



}// namespace generics
}// namespace rpy

#endif// ROUGHPY_GENERICS_INTERNAL_MONOMIAL_H
