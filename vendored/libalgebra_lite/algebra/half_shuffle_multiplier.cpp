//
// Created by sam on 03/09/22.
//

#include "libalgebra_lite/shuffle_tensor.h"
#include "libalgebra_lite/free_tensor.h"





using namespace lal;

namespace lal {
template class base_multiplier<left_half_shuffle_tensor_multiplier, tensor_basis>;
template class base_multiplier<right_half_shuffle_tensor_multiplier, tensor_basis>;


std::ostream& operator<<(std::ostream& os, std::pair<const tensor_basis*, index_key<>> arg)
{
    return arg.first->print_key(os, arg.second);
}


} // namespace lal


typename left_half_shuffle_tensor_multiplier::product_type
left_half_shuffle_tensor_multiplier::shuffle(
        const tensor_basis& basis, key_type lhs, key_type rhs) const
{
    const auto lhs_deg = lhs.degree();
    const auto rhs_deg = rhs.degree();

    if (lhs_deg == 0) {
        return {{rhs, 1}};
    }
    if (rhs_deg == 0) {
        return {{lhs, 1}};
    }

//    const auto& left = operator()(basis, lhs, rhs);
//    const auto& right = operator()(basis, rhs, lhs);
    return base_type::add(operator()(basis, lhs, rhs),
            operator()(basis, rhs, lhs));
//    return base_type::add(left, right);
}

typename left_half_shuffle_tensor_multiplier::product_type
lal::left_half_shuffle_tensor_multiplier::key_prod_impl(
        const tensor_basis& basis,
        key_type lhs, key_type rhs) const
{
    using ftm = free_tensor_multiplier;

    std::map<key_type, scalar_type> tmp;

    const auto lhs_deg = static_cast<deg_t>(lhs.degree());
    const auto rhs_deg = static_cast<deg_t>(rhs.degree());

    if (lhs_deg + rhs_deg <= basis.depth()) {

        if (lhs_deg == 0) {
            return {};
        }
        if (rhs_deg == 0) {
            return {{lhs, 1}};
        }

        const auto lparent = basis.lparent(lhs);
        const auto& right_part = shuffle(basis, basis.rparent(lhs), rhs);

        for (const auto& item : right_part) {
            tmp[ftm::concat_product(basis, lparent, item.first)] += item.second;
        }
    }

    return {tmp.begin(), tmp.end()};
}


typename left_half_shuffle_tensor_multiplier::reference
left_half_shuffle_tensor_multiplier::operator()(
        const tensor_basis& basis, key_type lhs, key_type rhs) const
{
    static const boost::container::small_vector<typename base_type::pair_type, 0> null;

    if (static_cast<deg_t>(lhs.degree() + rhs.degree()) > basis.depth()) {
        return null;
    }

    parent_type parents{lhs, rhs};
    auto found = m_cache.find(parents);
    if (found != m_cache.end()) {
        return found->second;
    }

    return m_cache[parents] = key_prod_impl(basis, lhs, rhs);
}

typename right_half_shuffle_tensor_multiplier::parent_type
right_half_shuffle_tensor_multiplier::split_at_right(
        const tensor_basis& basis, key_type key) const noexcept
{
    const auto width = basis.width();
    const auto deg = key.degree();
    const auto index = key.index();

    if (deg == 0) {
        return { key, key };
    }
    if (deg == 1) {
        const key_type null(0, 0);
        return { null, key };
    }

    return { key_type(key.degree()-1, index / width),
             key_type(1, index % width) };
}

typename right_half_shuffle_tensor_multiplier::product_type
right_half_shuffle_tensor_multiplier::key_prod_impl(
        const tensor_basis& basis,
        key_type lhs,
        key_type rhs) const
{

    std::map<key_type, scalar_type> tmp;

    const auto lhs_deg = static_cast<deg_t>(lhs.degree());
    const auto rhs_deg = static_cast<deg_t>(rhs.degree());

    if (lhs_deg + rhs_deg <= basis.depth()) {

        if (lhs_deg == 0) {
            return {};
        }
        if (rhs_deg == 0) {
            return {{lhs, 1}};
        }

        const auto parents = split_at_right(basis, rhs);
        const auto& left_part = shuffle(basis, lhs, parents.first);

        for (const auto& item : left_part) {
            tmp[free_tensor_multiplier::concat_product(basis, item.first, parents.second)] += item.second;
        }
    }

    return {tmp.begin(), tmp.end()};
}
typename right_half_shuffle_tensor_multiplier::product_type
right_half_shuffle_tensor_multiplier::shuffle(
        const tensor_basis& basis,
        key_type lhs, key_type rhs) const
{
    const auto lhs_deg = lhs.degree();
    const auto rhs_deg = rhs.degree();

    if (lhs_deg==0) {
        return {{rhs, 1}};
    }
    if (rhs_deg==0) {
        return {{lhs, 1}};
    }

    return base_type::add(operator()(basis, lhs, rhs),
            operator()(basis, rhs, lhs));
}

typename right_half_shuffle_tensor_multiplier::reference
right_half_shuffle_tensor_multiplier::operator()(
        const tensor_basis& basis,
        key_type lhs, key_type rhs) const
{
    static const boost::container::small_vector<typename base_type::pair_type, 0> null;

    if (lhs.degree() + rhs.degree() > static_cast<dimn_t>(basis.depth())) {
        return null;
    }

    parent_type parents{lhs, rhs};
    auto found = m_cache.find(parents);
    if (found != m_cache.end()) {
        return found->second;
    }

    return m_cache[parents] = key_prod_impl(basis, lhs, rhs);
}
