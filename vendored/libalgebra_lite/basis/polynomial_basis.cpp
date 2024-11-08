//
// Created by user on 05/09/22.
//

#include "libalgebra_lite/polynomial_basis.h"

#include <mutex>
#include <unordered_map>
#include <ostream>

namespace lal {


std::ostream& operator<<(std::ostream& os, const monomial& arg)
{
    bool first = true;
    for (const auto& item : arg) {
        if (first) {
            first = false;
        }
        else {
            os << ' ';
        }
        if (item.second > 0) {
            os << item.first;
            if (item.second > 1) {
                os << '^' << item.second;
            }
        }

    }
    return os;
}

monomial& monomial::operator*=(const monomial& rhs)
{
    const auto lend = m_data.end();
    for (const auto& item : rhs.m_data) {
        auto it = m_data.find(item.first);
        if (it == lend) {
            m_data.insert(item);
        } else {
            it->second += item.second;
        }
    }

    return *this;
}

monomial operator*(const monomial& lhs, const monomial& rhs)
{
    monomial result(lhs);
    result *= rhs;
    return result;
}


std::ostream& polynomial_basis::print_key(std::ostream& os, const polynomial_basis::key_type& key) const
{
    return os << key;
}

basis_pointer<polynomial_basis> basis_registry<polynomial_basis>::get()
{
    static const std::unique_ptr<const polynomial_basis> basis(new polynomial_basis);
    return basis_pointer<polynomial_basis>(basis);
}

}
