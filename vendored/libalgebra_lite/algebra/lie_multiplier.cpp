//
// Created by user on 27/08/22.
//


#include "libalgebra_lite/lie.h"

#include <mutex>
#include <unordered_map>

#include <boost/functional/hash.hpp>

namespace lal {

template class base_multiplier<lie_multiplier, hall_basis, 2>;

lie_multiplier::product_type
lie_multiplier::key_prod_impl(const hall_basis& basis, key_type lhs, key_type rhs) const
{
    assert(basis.width() == m_width);

    if (lhs>rhs) {
        return lie_multiplier::uminus(operator()(basis, rhs, lhs));
    }

    product_type result;
    if (basis.degree(lhs) + basis.degree(rhs) > basis.depth()) {
        return result;
    }

    auto found = basis.find(parent_type(lhs, rhs));
    if (found.found) {
        result.emplace_back(found.it->second, 1);
    } else {
        auto lparent = basis.lparent(rhs);
        auto rparent = basis.rparent(rhs);

        auto result_left = mul(basis, operator()(basis, lhs, lparent), rparent);
        auto result_right = mul(basis, operator()(basis, lhs, rparent), lparent);

        return sub(result_left, result_right);
    }

    return result;
}


lie_multiplier::reference lie_multiplier::operator()(
        const hall_basis& basis,
        lie_multiplier::key_type lhs, lie_multiplier::key_type rhs) const
{
    static const product_type null;
    if (lhs == rhs) {
        return null;
    }

    std::lock_guard<std::recursive_mutex> access(m_lock);

    parent_type parents {lhs, rhs};
    auto& found = m_cache[parents];
    if (!found.empty()) {
        return found;
    }

    found = key_prod_impl(basis, lhs, rhs);
    return found;
}



template class multiplication_registry<lie_multiplication>;

} // namespace alg
