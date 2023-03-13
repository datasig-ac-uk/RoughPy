#ifndef ROUGHPY_ALGEBRA_ALGEBRA_INFO_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_INFO_H_

#include "algebra_fwd.h"
#include <roughpy/config/traits.h>
#include <roughpy/scalars/scalar_type.h>

#include <boost/container/small_vector.hpp>

namespace rpy {
namespace algebra {

template <typename Basis>
struct basis_info {
    using this_key_type = typename Basis::key_type;

    static const BasisInterface* make(const Basis* basis) {
        return new BasisImplementation<Basis>(basis);
    }
    static this_key_type convert_key(const Basis &basis, rpy::key_type key) {
        return basis.index_to_key(key);
    }
    static rpy::key_type convert_key(const Basis &basis, const this_key_type &key) {
        return basis.key_to_index(key);
    }
    static rpy::key_type first_key(const Basis &basis) {
        return 0;
    }
    static rpy::key_type last_key(const Basis &basis) {
        return basis.size();
    }
    static deg_t native_degree(const Basis &basis, const this_key_type &key) {
        return basis.degree(key);
    }
    static deg_t degree(const Basis &basis, rpy::key_type key) {
        return native_degree(basis, convert_key(basis, key));
    }
};

template <typename Algebra>
struct algebra_info {
    using basis_type = typename Algebra::basis_type;
    using basis_traits = basis_info<basis_type>;
    using scalar_type = typename Algebra::scalar_type;
    using rational_type = scalar_type;
    using reference = scalar_type &;
    using const_reference = const scalar_type &;
    using pointer = scalar_type *;
    using const_pointer = const scalar_type *;

    static const scalars::ScalarType *ctype() noexcept { return scalars::ScalarType::of<scalar_type>(); }
    static constexpr VectorType vtype() noexcept { return VectorType::Sparse; }
    static deg_t width(const Algebra *instance) noexcept { return instance->basis().width(); }
    static deg_t max_depth(const Algebra *instance) noexcept { return instance->basis().depth(); }

    static const basis_type &basis(const Algebra &instance) noexcept { return instance.basis(); }

    using this_key_type = typename Algebra::key_type;
    static this_key_type convert_key(const Algebra *instance, rpy::key_type key) noexcept { return basis_traits::convert_key(instance->basis(), key); }

    static deg_t degree(const Algebra &instance) noexcept { return instance.degree(); }
    static deg_t degree(const Algebra *instance, rpy::key_type key) noexcept { return basis_traits::degree(instance->basis(), key); }
    static deg_t native_degree(const Algebra *instance, this_key_type key) { return basis_traits::native_degree(instance->basis(), key); }

    static key_type first_key(const Algebra *) noexcept { return 0; }
    static key_type last_key(const Algebra *instance) noexcept { return basis_traits::last_key(instance->basis()); }
//
//    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;
//    static const key_prod_container &key_product(const Algebra *inst, key_type k1, key_type k2) {
//        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
//        return null;
//    }
//    static const key_prod_container &key_product(const Algebra *inst, const this_key_type &k1, const this_key_type &k2) {
//        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
//        return null;
//    }

    static Algebra create_like(const Algebra &instance) {
        return Algebra();
    }
};
}
}
#endif // ROUGHPY_ALGEBRA_ALGEBRA_INFO_H_
