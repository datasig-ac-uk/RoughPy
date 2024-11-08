//
// Created by user on 27/07/22.
//
#include "libalgebra_lite/tensor_basis.h"

#include <algorithm>
#include <functional>

namespace lal {


tensor_basis::tensor_basis(deg_t width, deg_t depth) :
        m_width (width), m_depth(depth)
{
    m_powers.reserve(depth+1);
    m_sizes.reserve(depth+2);
    m_powers.push_back(1);
    m_sizes.push_back(1);

    for (deg_t d = 1; d<=depth; ++d) {
        m_powers.push_back(m_powers.back()*static_cast<dimn_t>(width));
        m_sizes.push_back(1+static_cast<dimn_t>(width)*m_sizes.back());
    }
    m_sizes.push_back(1+static_cast<dimn_t>(width)*m_sizes.back());
}

dimn_t tensor_basis::key_to_index(tensor_basis::key_type arg) const noexcept
{
    return start_of_degree(deg_t(arg.degree())) + arg.index();
}
tensor_basis::key_type tensor_basis::index_to_key(dimn_t arg) const noexcept
{
    if (arg == 0) {
        return key_type(0, 0);
    }

    auto it = std::lower_bound(m_sizes.begin(),
            m_sizes.end(), arg, std::less_equal<>());

    if (it == m_sizes.end()) {
        return key_type(0, 0);
    }

    auto degree = static_cast<deg_t>(it - m_sizes.begin());
    return key_type(degree, arg - *(--it));
}
dimn_t tensor_basis::size_of_degree(deg_t deg) const noexcept
{
    return m_powers[deg];
}

std::string tensor_basis::key_to_string(const tensor_basis::key_type& key) const
{
    std::stringstream ss;
    print_key(ss, key);
    return ss.str();
}
std::ostream& tensor_basis::print_key(std::ostream& os, const tensor_basis::key_type& key) const
{
    std::vector<let_t> tmp;
    auto deg = static_cast<deg_t>(key.degree());
    tmp.reserve(deg);
    auto idx = key.index();
    for (deg_t i=0; i<deg; ++i) {
        tmp.push_back(1 + (idx % m_width));
        idx /= m_width;
    }

    bool first = true;
    auto end = tmp.crend();
    for (auto it = tmp.crbegin(); it != end; ++it) {
        if (first) {
            first = false;
        } else {
            os << ',';
        }
        os << *it;
    }
    return os;
}

void tensor_basis::advance_key(tensor_basis::key_type &key) const noexcept {
    const auto degree = static_cast<deg_t>(key.degree());
    ++key;
    if (key.index() >= m_powers[degree]) {
        key = key_type(degree+1, 0);
    }
}

template class basis_registry<tensor_basis>;

}
