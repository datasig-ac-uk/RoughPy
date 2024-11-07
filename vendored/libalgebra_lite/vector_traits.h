//
// Created by user on 27/07/22.
//

#ifndef LIBALGEBRA_LITE_VECTOR_TRAITS_H
#define LIBALGEBRA_LITE_VECTOR_TRAITS_H

namespace lal {
namespace dtl {

template<typename Vector>
struct vector_traits {
    using basis_type = typename Vector::basis_type;

    using coefficient_ring = typename Vector::coefficient_ring;
    using scalar_type = typename Vector::scalar_type;
    using rational_type = typename Vector::rational_type;
};

} // namespace dtl
} // namespace lal

#endif //LIBALGEBRA_LITE_VECTOR_TRAITS_H
