//
// Created by user on 07/02/23.
//

#ifndef LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_VECTOR_BASE_H
#define LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_VECTOR_BASE_H

#include "implementation_types.h"



#include "basis.h"
#include "coefficients.h"
#include "basis_traits.h"


namespace lal {
namespace vectors {

template <typename Basis, typename Coefficients>
class vector_base
{
protected:
    using coeff_traits = coefficient_trait<Coefficients>;
    using basis_traits = basis_trait<Basis>;

    using basis_pointer = lal::basis_pointer<Basis>;
    basis_pointer p_basis;

    explicit vector_base(basis_pointer basis) : p_basis(basis) {}
public:

    using basis_type            = Basis;
    using key_type              = typename basis_traits::key_type;
    using coefficient_ring      = typename coeff_traits::coefficient_ring;
    using scalar_type           = typename coeff_traits::scalar_type;
    using rational_type         = typename coeff_traits::rational_type;

    basis_pointer get_basis() const noexcept { return p_basis; }
    const Basis& basis() const noexcept { return *p_basis; }

    void swap(vector_base& other) {
        std::swap(p_basis, other.p_basis);
    }

};


} // namespace vectors
} // namespace lal



#endif //LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_VECTOR_BASE_H
