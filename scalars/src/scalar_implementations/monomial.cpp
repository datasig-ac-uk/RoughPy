//
// Created by sam on 3/26/24.
//

#include "types/monomial.h"

using namespace rpy;
using namespace rpy::scalars;

namespace {
std::ostream& operator<<(std::ostream& os, const Monomial::letter_type& letter)
{
    os << letter.packed();
    const auto id = letter.base();
    if (RPY_LIKELY(id != 0)) { os << id; }
    return os;
}
}// namespace

deg_t Monomial::degree() const noexcept
{
    return ranges::accumulate(m_data | views::values, 0, std::plus<>());
}
Monomial& Monomial::operator*=(const Monomial& rhs)
{
    for (const auto& [key, value] : rhs.m_data) { m_data[key] += value; }
    return *this;
}
Monomial& Monomial::operator*=(letter_type rhs)
{
    m_data[rhs] += 1;
    return *this;
}
std::ostream& scalars::operator<<(std::ostream& os, const Monomial& arg)
{
    bool first = true;
    for (const auto& [key, deg] : arg) {
        if (first) {
            first = false;
        } else {
            os << ' ';
        }
        os << key;
        if (deg > 1) { os << '^' << deg; }
    }
    return os;
}
