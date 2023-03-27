//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LIE_INFO_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LIE_INFO_H

#include "algebra_info.h"
#include "lie_basis_info.h"

#include <roughpy/core/traits.h>
#include <roughpy/scalars/scalar_type.h>

#include <libalgebra_lite/lie.h>

#include "vector_type_helper.h"


namespace rpy { namespace algebra {

template <typename Coeffs, template <typename, typename> class VectorType,
          template <typename> class Storage>
struct algebra_info<lal::lie<Coeffs, VectorType, Storage>> {
    using basis_type = lal::hall_basis;
    using basis_traits = basis_info<lal::hall_basis>;

    using algebra_t = lal::lie<Coeffs, VectorType, Storage>;

    using scalar_type = typename algebra_t::scalar_type;
    using rational_type = typename algebra_t::rational_type;
    using reference = decltype(std::declval<algebra_t&>()[std::declval<typename basis_type::key_type>()]);
    using const_reference = decltype(std::declval<const algebra_t&>()[std::declval<typename basis_type::key_type>()]);

    static const scalars::ScalarType *ctype() noexcept {
        return scalars::ScalarType::of<scalar_type>();
    }
    static constexpr rpy::algebra::VectorType vtype() noexcept {
        return dtl::vector_type_helper<VectorType>::vtype;
    }
    static deg_t width(const algebra_t *instance) noexcept {
        return instance->basis().width();
    }

    static deg_t max_depth(const algebra_t *instance) noexcept {
        return instance->basis().depth();
    }

    static const basis_type &basis(const algebra_t &instance) noexcept {
        return instance.basis();
    }

    using this_key_type = typename algebra_t::key_type;
    static this_key_type convert_key(const algebra_t *instance, rpy::key_type key) noexcept {
        return basis_traits::convert_key(instance->basis(), key);
    }

    static deg_t degree(const algebra_t &instance) noexcept {
        return instance.degree();
    }
    static deg_t degree(const algebra_t *instance, rpy::key_type key) noexcept {
        return basis_traits::degree(instance->basis(), key);
    }
    static deg_t native_degree(const algebra_t *instance, this_key_type key) {
        return basis_traits::native_degree(instance->basis(), key);
    }

    static key_type first_key(const algebra_t *) noexcept {
        return 1;
    }
    static key_type last_key(const algebra_t *instance) noexcept {
        return basis_traits::last_key(instance->basis());
    }

//    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;
//    static const key_prod_container &key_product(const algebra_t *inst, key_type k1, key_type k2) {
//        return key_product(inst, convert_key(inst, k1), convert_key(inst, k2));
//    }
//    static const key_prod_container &key_product(const algebra_t *inst, const this_key_type &k1, const this_key_type &k2) {
//
//    }

    static algebra_t create_like(const algebra_t &instance) {
        return algebra_t(instance.get_basis(), instance.multiplication());
    }
};

}}


#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LIE_INFO_H
