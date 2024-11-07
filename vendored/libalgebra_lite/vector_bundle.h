//
// Created by user on 24/08/22.
//

#ifndef LIBALGEBRA_LITE_VECTOR_BUNDLE_H
#define LIBALGEBRA_LITE_VECTOR_BUNDLE_H

#include <utility>

namespace lal {

template <typename Vector, typename Fibre=Vector>
class vector_bundle : public Vector
{
    Fibre m_fibre;

public:
    using vector_type = Vector;
    using fibre_type = Fibre;

    using fibre_basis_type = typename Fibre::basis_type;
    using fibre_key_type = typename Fibre::key_type;
    using fibre_coeffificient_ring = typename Fibre::coefficient_ring;
    using fibre_scalar_type = typename Fibre::scalar_type;
    using fibre_rational_type = typename Fibre::rational_type;
    using fibre_iterator = typename Fibre::iterator;
    using fibre_const_iterator = typename Fibre::const_iterator;
    using fibre_reference = typename Fibre::reference;
    using fibre_const_reference = typename Fibre::const_reference;


private:

    using vector_basis_pointer = const typename vector_bundle::basis_type*;
    using fibre_basis_pointer = const typename vector_bundle::fibre_basis_type*;

    vector_bundle(
            vector_basis_pointer vbasis,
            typename Vector::vector_type&& varg,
            fibre_basis_pointer fbasis,
            typename Fibre::vector_type&& farg
            )
        : Vector(vbasis, std::move(varg)), m_fibre(fbasis, farg)
    {}

public:

    explicit vector_bundle(vector_type&& arg)
        : Vector(std::move(arg)), m_fibre()
    {}

    explicit vector_bundle(const vector_type& varg)
        : Vector(varg), m_fibre()
    {}

    vector_bundle(vector_type&& varg, fibre_type&& farg)
        : Vector(std::move(varg)), m_fibre(std::move(farg))
    {}







};




} // namespace alg



#endif //LIBALGEBRA_LITE_VECTOR_BUNDLE_H
