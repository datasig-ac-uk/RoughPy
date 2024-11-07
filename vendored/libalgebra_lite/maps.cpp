//
// Created by user on 05/09/22.
//

#include "libalgebra_lite/maps.h"


#include <mutex>
#include <unordered_map>

#include <boost/functional/hash.hpp>

using namespace lal;

maps::maps(basis_pointer<tensor_basis> tbasis, basis_pointer<hall_basis> lbasis)
   : p_tensor_basis(std::move(tbasis)), p_lie_basis(std::move(lbasis)),
   p_impl(new dtl::maps_implementation(p_tensor_basis, p_lie_basis))
{

}

maps::~maps() {
    delete p_impl;
}

dtl::generic_commutator::tensor_type dtl::generic_commutator::operator()(
        ref_type lhs,
        ref_type rhs) const
{
    std::map<key_type, int> tmp;

    for (const auto& lhs_val : lhs) {
        for (const auto& rhs_val : rhs) {
            for (const auto& inner : m_mul.multiply(m_basis, lhs_val.first, rhs_val.first)) {
                tmp[inner.first] += inner.second*lhs_val.second*rhs_val.second;
            }
            for (const auto& inner : m_mul.multiply(m_basis, rhs_val.first, lhs_val.first)) {
                tmp[inner.first] -= inner.second*rhs_val.second*lhs_val.second;
            }
        }
    }
    return {tmp.begin(), tmp.end()};
}

typename dtl::maps_implementation::generic_tensor dtl::maps_implementation::expand_letter(let_t letter)
{
    return {{ tkey_type(1, letter-1), 1 }};
}
typename dtl::maps_implementation::glie_ref dtl::maps_implementation::rbracketing(
        dtl::maps_implementation::tkey_type tkey) const
{
    static const boost::container::small_vector<std::pair<tkey_type, int>, 0> null;

    if (tkey.degree()==0) {
        return null;
    }

    std::lock_guard<std::recursive_mutex> access(m_rbracketing_lock);

    auto& found = m_rbracketing_cache[tkey];
    if (found.set) {
        return found.value;
    }

    found.set = true;
    if (p_tensor_basis->letter(tkey)) {
        found.value.emplace_back(lkey_type(1, tkey.index()), 1);
        return found.value;
    }

    auto lhs = p_tensor_basis->lparent(tkey);
    auto rhs = p_tensor_basis->rparent(tkey);

    return found.value = p_lie_mul->multiply_generic(*p_lie_basis, rbracketing(lhs), rbracketing(rhs));
}
typename dtl::maps_implementation::gtensor_ref dtl::maps_implementation::expand(
        dtl::maps_implementation::lkey_type lkey) const
{
    return m_expand(lkey);
}

maps::maps(deg_t width, deg_t depth)
    : p_tensor_basis(basis_registry<tensor_basis>::get(width, depth)),
      p_lie_basis(basis_registry<hall_basis>::get(width, depth))
{
    static std::mutex lock;
    static std::unordered_map<std::pair<deg_t, deg_t>, std::unique_ptr<const dtl::maps_implementation>,
                              boost::hash<std::pair<deg_t, deg_t>>> cache;

    std::lock_guard<std::mutex> access(lock);
    auto& found = cache[{width, depth}];
    if (found) {
        p_impl = found.get();
    }

    found = std::make_unique<const dtl::maps_implementation>(p_tensor_basis, p_lie_basis);
    p_impl = found.get();
}
