//
// Created by user on 12/08/22.
//

#ifndef LIBALGEBRA_LITE_LIE_H
#define LIBALGEBRA_LITE_LIE_H

#include "implementation_types.h"
#include "libalgebra_lite_export.h"

#include <mutex>
#include <unordered_map>

#include <boost/container/small_vector.hpp>
#include <boost/functional/hash.hpp>

#include "algebra.h"
#include "hall_set.h"
#include "registry.h"

namespace lal {

class lie_multiplier;

LAL_EXPORT_TEMPLATE_CLASS(base_multiplier, lie_multiplier, hall_basis, 2)

class LIBALGEBRA_LITE_EXPORT lie_multiplier
    : public base_multiplier<lie_multiplier, hall_basis, 2>
{
    using base_type = base_multiplier<lie_multiplier, hall_basis, 2>;

    deg_t m_width;

    using typename base_type::key_type;
    using typename base_type::product_type;
    using typename base_type::reference;

    using parent_type = std::pair<key_type, key_type>;

    mutable std::unordered_map<
            parent_type, product_type, boost::hash<parent_type>>
            m_cache;
    mutable std::recursive_mutex m_lock;

    product_type
    key_prod_impl(const hall_basis& basis, key_type lhs, key_type rhs) const;

public:
    using basis_type = hall_basis;

    explicit lie_multiplier(deg_t width) : m_width(width) {}

    reference
    operator()(const hall_basis& basis, key_type lhs, key_type rhs) const;
};

struct LIBALGEBRA_LITE_EXPORT lie_multiplication
    : public base_multiplication<lie_multiplier> {
    using base = base_multiplication<lie_multiplier>;
    using base::base;
};

template <
        typename Coefficients, template <typename, typename> class VectorType,
        template <typename> class StorageModel>
using lie = algebra<
        hall_basis, Coefficients, lie_multiplication, VectorType, StorageModel>;

LAL_EXPORT_TEMPLATE_CLASS(multiplication_registry, lie_multiplication)

template <typename LLie, typename RLie>
inline LLie lie_multiply(const LLie& left, const RLie& right)
{
    const auto liem
            = multiplication_registry<lie_multiplication>::get(left.basis());
    return multiply(*liem, left, right);
}

}// namespace lal

#endif// LIBALGEBRA_LITE_LIE_H
