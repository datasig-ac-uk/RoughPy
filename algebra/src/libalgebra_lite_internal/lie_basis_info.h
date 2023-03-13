//
// Created by user on 07/03/23.
//

#include "basis.h"
#include "basis_impl.h"
#include "algebra_info.h"


#include <libalgebra_lite/hall_set.h>

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LIE_BASIS_INFO_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LIE_BASIS_INFO_H

namespace rpy { namespace algebra {

using ll_lkey_type = typename lal::hall_basis::key_type;

template <>
const BasisInterface* basis_info<lal::hall_basis>::make(const lal::hall_basis* basis) {
    return new BasisImplementation<lal::hall_basis>(basis);
}

template <>
ll_lkey_type basis_info<lal::hall_basis>::convert_key(const lal::hall_basis &basis, rpy::key_type key) {
    return basis.index_to_key(key-1);
}

template <>
rpy::key_type basis_info<lal::hall_basis>::convert_key(const lal::hall_basis &basis, const this_key_type &key) {
    return basis.key_to_index(key) + 1;
}

template <>
rpy::key_type basis_info<lal::hall_basis>::first_key(const lal::hall_basis &basis) {
    return 1;
}

template <>
rpy::key_type basis_info<lal::hall_basis>::last_key(const lal::hall_basis &basis) {
    return basis.size(-1) + 1;
}

template <>
deg_t basis_info<lal::hall_basis>::native_degree(const lal::hall_basis &basis, const this_key_type &key) {
    return key.degree();
}

template <>
deg_t basis_info<lal::hall_basis>::degree(const lal::hall_basis &basis, rpy::key_type key) {
    return basis.index_to_key(key-1).degree();
}

}}


#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LIE_BASIS_INFO_H
