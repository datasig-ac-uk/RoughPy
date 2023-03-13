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
struct basis_info<lal::hall_basis> {
    using this_key_type = ll_lkey_type;
    static const BasisInterface * make(const lal::hall_basis *basis) {
        return new BasisImplementation<lal::hall_basis>(basis);
    }
    static ll_lkey_type convert_key(const lal::hall_basis &basis, rpy::key_type key) {
        return basis.index_to_key(key - 1);
    }
    static rpy::key_type convert_key(const lal::hall_basis &basis, const this_key_type &key) {
        return basis.key_to_index(key) + 1;
    }
    static rpy::key_type first_key(const lal::hall_basis &basis) {
        return 1;
    }
    static rpy::key_type last_key(const lal::hall_basis &basis) {
        return basis.size(-1) + 1;
    }
    static deg_t native_degree(const lal::hall_basis &basis, const this_key_type &key) {
        return key.degree();
    }
    static deg_t degree(const lal::hall_basis &basis, rpy::key_type key) {
        return basis.index_to_key(key - 1).degree();
    }
};


}}


#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LIE_BASIS_INFO_H
