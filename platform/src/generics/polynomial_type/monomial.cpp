//
// Created by sam on 26/11/24.
//

#include "monomial.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <ostream>

#include "roughpy/core/ranges.h"


using namespace rpy;
using namespace rpy::generics;

deg_t Monomial::degree() const noexcept
{
    return std::accumulate(m_data.begin(),
                           m_data.end(),
                           0,
                           [](auto acc, const auto& pair) {
                               return acc + pair.second;
                           });
}


deg_t Monomial::operator[](Indeterminate indeterminate) const noexcept {
    auto it = m_data.find(indeterminate);
    if (it == m_data.end()) { return 0; }
    return it->second;
}

Monomial& Monomial::operator*=(const Monomial& rhs)
{
    const auto lend = m_data.end();
    for (const auto& [ind, pow] : rhs) {
        if (auto it = m_data.find(ind); it != lend) {
            it->second += pow;
        } else { m_data.emplace(ind, pow); }
    }
    return *this;
}

Monomial generics::operator*(const Monomial& lhs, const Monomial& rhs)
{
    Monomial result(lhs);
    result *= rhs;
    return result;
}

std::ostream& generics::operator<<(std::ostream& os, const Monomial& value)
{
    bool first = true;
    for (const auto& [ind, pow] : value) {
        if (first) { first = false; } else { os << ' '; }
        if (pow > 0) {
            os << ind;
            if (pow > 1) { os << '^' << pow; }
        }
    }
    return os;
}

bool generics::operator==(const Monomial& lhs, const Monomial& rhs) noexcept
{
    if (lhs.type() != rhs.type()) { return false; }

    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool generics::operator<(const Monomial& lhs, const Monomial& rhs) noexcept
{
    const auto ldegree = lhs.degree();
    const auto rdegree = rhs.degree();

    if (ldegree < rdegree) { return true; }

    if (ldegree == rdegree) {
        return std::lexicographical_compare(lhs.begin(),
                                            lhs.end(),
                                            rhs.begin(),
                                            rhs.end());
    }

    return false;
}

bool generics::operator<=(const Monomial& lhs, const Monomial& rhs) noexcept
{
    const auto ldegree = lhs.degree();
    const auto rdegree = rhs.degree();

    if (ldegree < rdegree) { return true; }

    if (ldegree == rdegree) {
        return std::lexicographical_compare(lhs.begin(),
                                            lhs.end(),
                                            rhs.begin(),
                                            rhs.end(),
                                            std::less_equal<>());
    }

    return false;
}



hash_t generics::hash_value(const Monomial& value)
{
    Hash<Monomial::map_type> hasher;
    return hasher(value.m_data);
}
