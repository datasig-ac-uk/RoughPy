//
// Created by user on 04/09/22.
//

#include "libalgebra_lite/free_tensor.h"


using namespace lal;




typename free_tensor_multiplier::product_type
free_tensor_multiplier::operator()(
        const tensor_basis& basis,
        free_tensor_multiplier::key_type lhs,
        free_tensor_multiplier::key_type rhs) const
{
    // Note that product_type is small and the product only ever
    // has 1 element (if any) so no allocation takes place
    if (lhs.degree() + rhs.degree() <= static_cast<dimn_t>(basis.depth())) {
        return product_type{{concat_product(basis, lhs, rhs), 1}};
    }
    return {};
}


namespace lal {
template class multiplication_registry<free_tensor_multiplication>;
}
