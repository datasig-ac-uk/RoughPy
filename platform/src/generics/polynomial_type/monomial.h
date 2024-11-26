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
            small_vector<pair<const Indeterminate, deg_t>, 1>;
    using map_type = boost::container::
            flat_map<Indeterminate, deg_t, std::less<>, map_storage_type>;

    map_type m_data;

public:

    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;

    Monomial() = default;

    explicit Monomial(Indeterminate indeterminate, deg_t degree=1) noexcept
    {
        m_data.emplace(indeterminate, degree);
    }

    template <typename InputIt>
    explicit Monomial(InputIt begin, InputIt end)
        : m_data(begin, end)
    {}


    deg_t degree() const noexcept;
    deg_t type() const noexcept { return m_data.size(); }

    iterator begin() noexcept { return m_data.begin(); }
    iterator end() noexcept { return m_data.end(); }
    const_iterator begin() const noexcept { return m_data.begin(); }
    const_iterator end() const noexcept { return m_data.end(); }


    Monomial& operator*=(const Monomial& rhs);


    friend hash_t hash_value(const Monomial& value);

};


hash_t hash_value(const Monomial& value);
Monomial operator*(const Monomial& lhs, const Monomial& rhs);
std::ostream &operator<<(std::ostream &os, const Monomial& value);



}// namespace generics
}// namespace rpy

#endif// ROUGHPY_GENERICS_INTERNAL_MONOMIAL_H
