//
// Created by user on 01/09/22.
//


#include <libalgebra_lite/polynomial_basis.h>
#include <libalgebra_lite/packed_integer.h>

using namespace lal;

template class dtl::packed_integer<dimn_t, char>;

deg_t lal::monomial::operator[](letter_type let) const noexcept {
    auto found = m_data.find(let);
    if (found != m_data.end()) {
        return found->second;
    }
    return 0;
}
deg_t monomial::degree() const noexcept
{
    return std::accumulate(
            m_data.begin(), m_data.end(), 0,
            [](const auto& curr, const auto& key) { return curr + key.second; }
    );
}
