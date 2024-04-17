//
// Created by sam on 3/26/24.
//

#include "types/monomial.h"
#include "scalar_serialization.h"

using namespace rpy;
using namespace rpy::scalars;

namespace rpy {
namespace scalars {
std::ostream& operator<<(std::ostream& os, const indeterminate_type& letter)
{
    os << letter.packed();
    const auto id = letter.base();
    if (RPY_LIKELY(id != 0)) { os << id; }
    return os;
}
}// namespace scalars
}// namespace rpy

deg_t Monomial::degree() const noexcept
{
    return ranges::accumulate(m_data | views::values, 0, std::plus<>());
}
Monomial& Monomial::operator*=(const Monomial& rhs)
{
    for (const auto& [key, value] : rhs.m_data) { m_data[key] += value; }
    return *this;
}
Monomial& Monomial::operator*=(indeterminate_type rhs)
{
    m_data[rhs] += 1;
    return *this;
}
std::ostream& scalars::operator<<(std::ostream& os, const Monomial& arg)
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

// #define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::indeterminate_type
// #include <roughpy/platform/serialization_instantiations.inl>
//
// #define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::Monomial
// #include <roughpy/platform/serialization_instantiations.inl>
