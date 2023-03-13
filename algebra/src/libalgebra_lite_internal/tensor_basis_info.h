//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_BASIS_INFO_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_BASIS_INFO_H

#include "basis.h"
#include "basis_impl.h"
#include "algebra_info.h"

#include <libalgebra_lite/tensor_basis.h>

namespace rpy {
namespace algebra {

using ll_tkey_type = typename lal::tensor_basis::key_type;

template <>
const BasisInterface* basis_info<lal::tensor_basis>::make(const lal::tensor_basis* basis) {
    return new BasisImplementation<lal::tensor_basis>(basis);
}

template <>
ll_tkey_type basis_info<lal::tensor_basis>::convert_key(const lal::tensor_basis& basis, rpy::key_type key) {
    return basis.index_to_key(key);
}

template <>
rpy::key_type basis_info<lal::tensor_basis>::convert_key(const lal::tensor_basis &basis, const this_key_type& key) {
    return basis.key_to_index(key);
}

template <>
rpy::key_type basis_info<lal::tensor_basis>::first_key(const lal::tensor_basis &basis) {
    return 0;
}

template <>
rpy::key_type basis_info<lal::tensor_basis>::last_key(const lal::tensor_basis &basis) {
    return basis.size(-1);
}

template <>
deg_t basis_info<lal::tensor_basis>::native_degree(const lal::tensor_basis &basis, const this_key_type &key) {
    return key.degree();
}

template <>
deg_t basis_info<lal::tensor_basis>::degree(const lal::tensor_basis& basis, rpy::key_type key) {
    return basis.index_to_key(key).degree();
}


}// namespace algebra
}// namespace rpy

#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_BASIS_INFO_H
