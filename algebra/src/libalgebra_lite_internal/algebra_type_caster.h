//
// Created by user on 18/07/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_CASTER_H_
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_CASTER_H_

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/algebra/algebra_impl.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/free_tensor_impl.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/shuffle_tensor.h>

#include "algebra_type_helper.h"
#include "lite_vector_selector.h"
#include "vector_type_helper.h"

namespace rpy {
namespace algebra {

template <AlgebraType Type>
struct algebra_type_selector;

template <>
struct algebra_type_selector<AlgebraType::FreeTensor> {
    template <typename C, VectorType V>
    using type = typename dtl::vector_type_selector<V>::template free_tensor<C>;
};
template <>
struct algebra_type_selector<AlgebraType::ShuffleTensor> {
    template <typename C, VectorType V>
    using type =
            typename dtl::vector_type_selector<V>::template shuffle_tensor<C>;
};
template <>
struct algebra_type_selector<AlgebraType::Lie> {
    template <typename C, VectorType V>
    using type = typename dtl::vector_type_selector<V>::template lie<C>;
};
template <>
struct algebra_type_selector<AlgebraType::FreeTensorBundle> {
    template <typename C, VectorType V>
    using type = typename dtl::vector_type_selector<V>::template free_tensor<C>;
};
template <>
struct algebra_type_selector<AlgebraType::ShuffleTensorBundle> {
    template <typename C, VectorType V>
    using type =
            typename dtl::vector_type_selector<V>::template shuffle_tensor<C>;
};
template <>
struct algebra_type_selector<AlgebraType::LieBundle> {
    template <typename C, VectorType V>
    using type = typename dtl::vector_type_selector<V>::template lie<C>;
};

template <
        typename Coefficients, typename AlgebraTag = void,
        typename VectorTag = void>
struct algebra_type_caster;

template <typename Coefficients>
struct algebra_type_caster<Coefficients, void, void> {
    template <AlgebraType AType>
    using refine
            = algebra_type_caster<Coefficients, algebra_type_tag<AType>, void>;
};

template <typename Coefficients, typename ATag>
struct algebra_type_caster<Coefficients, ATag, void> {
    template <VectorType VType>
    using refine
            = algebra_type_caster<Coefficients, ATag, vector_type_tag<VType>>;
};

template <typename Coefficients, AlgebraType AType, VectorType VType>
struct algebra_type_caster<
        Coefficients, algebra_type_tag<AType>, vector_type_tag<VType>> {

    using type = typename algebra_type_selector<AType>::template type<
            Coefficients, VType>;
    using info = dtl::alg_details_of<type>;
    using interface_t = typename info::interface_type;

    static type& cast(RawUnspecifiedAlgebraType arg)
    {
        RPY_CHECK(arg != nullptr);
        RPY_CHECK(arg->alg_type() == AType);
        RPY_CHECK(arg->storage_type() == VType);
        auto* interface_ptr = reinterpret_cast<interface_t*>(arg);
        return algebra_cast<type>(*interface_ptr);
    }
    static const type& cast(ConstRawUnspecifiedAlgebraType arg) {
        RPY_CHECK(arg != nullptr);
        RPY_CHECK(arg->alg_type() == AType);
        RPY_CHECK(arg->storage_type() == VType);
        const auto* interface_ptr = reinterpret_cast<const interface_t*>(arg);
        return algebra_cast<type>(*interface_ptr);
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_CASTER_H_
