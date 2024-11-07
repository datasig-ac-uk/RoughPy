//
// Created by user on 30/08/22.
//

#ifndef LIBALGEBRA_LITE_POLYNOMIAL_H
#define LIBALGEBRA_LITE_POLYNOMIAL_H

#include "implementation_types.h"
#include "libalgebra_lite_export.h"

#include <boost/container/small_vector.hpp>

#include "algebra.h"
#include "coefficients.h"
#include "polynomial_basis.h"
#include "registry.h"
#include "sparse_vector.h"

namespace lal {

class LIBALGEBRA_LITE_EXPORT polynomial_multiplier
{
    using letter_type = typename polynomial_basis::letter_type;
    using key_type = typename polynomial_basis::key_type;

    using product_type
            = boost::container::small_vector<std::pair<key_type, int>, 1>;

public:
    using basis_type = polynomial_basis;

    product_type operator()(
            const polynomial_basis& basis, const key_type& lhs,
            const key_type& rhs
    ) const;
};

template <>
class LIBALGEBRA_LITE_EXPORT
        multiplication_registry<base_multiplication<polynomial_multiplier>>
{
    using multiplication = base_multiplication<polynomial_multiplier>;

public:
    static std::shared_ptr<const multiplication> get();
    template <typename Basis>
    static std::shared_ptr<const multiplication> get(const Basis&)
    {
        return get();
    }
};

template <typename Coefficients>
class polynomial : public algebra<
                           polynomial_basis, Coefficients,
                           base_multiplication<polynomial_multiplier>,
                           sparse_vector, dtl::standard_storage>
{
    using base = algebra<
            polynomial_basis, Coefficients,
            base_multiplication<polynomial_multiplier>, sparse_vector,
            dtl::standard_storage>;

public:
    using multiplication_type = base_multiplication<polynomial_multiplier>;
    using base::base;

    polynomial() : base(basis_registry<polynomial_basis>::get()) {}

    template <typename Scalar>
    explicit polynomial(Scalar s)
        : base(basis_registry<polynomial_basis>::get(),
               multiplication_registry<multiplication_type>::get(), monomial(),
               typename base::scalar_type(s))
    {}

    template <typename Key, typename Scalar>
    explicit polynomial(Key k, Scalar s)
        : base(basis_registry<polynomial_basis>::get(),
               multiplication_registry<multiplication_type>::get())
    {
        (*this)[typename base::key_type(k)] = typename base::scalar_type(s);
    }

    template <typename IndeterminateMap>
    typename polynomial::scalar_type operator()(const IndeterminateMap& arg
    ) const noexcept
    {
        using ring = typename polynomial::coefficient_ring;
        auto ans = ring::zero();
        for (const auto& item : *this) {
            auto key_result = item.first.template eval<ring>(arg);
            ring::add_inplace(ans, ring::mul(item.second, key_result));
        }
        return ans;
    }
};

LAL_EXPORT_TEMPLATE_CLASS(polynomial, double_field)
LAL_EXPORT_TEMPLATE_CLASS(polynomial, float_field)
using double_poly = polynomial<double_field>;
using float_poly = polynomial<float_field>;

#ifdef LAL_ENABLE_RATIONAL_COEFFS
LAL_EXPORT_TEMPLATE_CLASS(polynomial, rational_field)
using rational_poly = polynomial<rational_field>;
#endif




template <typename Field>
struct coefficient_ring<polynomial<Field>, typename Field::rational_type>
{
    using scalar_type = polynomial<Field>;
    using rational_type = typename Field::rational_type;

    static const scalar_type& zero() noexcept
    {
        static const scalar_type zero;
        return zero;
    }
    static const scalar_type& one() noexcept
    {
        static const scalar_type one(1);
        return one;
    }
    static const scalar_type& mone() noexcept
    {
        static const scalar_type mone(-1);
        return mone;
    }

    static inline bool is_invertible(const scalar_type& arg) {
        return arg.size() == 1
                && arg.degree() == 0
                && Field::is_invertible(arg.begin()->value());
    }
    static constexpr const rational_type& as_rational(const scalar_type& arg)
            noexcept {
        return Field::as_rational(arg.begin()->value());
    }

};

LAL_EXPORT_TEMPLATE_STRUCT(coefficient_ring, double_poly, double)
LAL_EXPORT_TEMPLATE_STRUCT(coefficient_ring, float_poly, float)

#ifdef LAL_ENABLE_RATIONAL_COEFFS
LAL_EXPORT_TEMPLATE_STRUCT(
        coefficient_ring, rational_poly, typename rational_field::scalar_type
)

using polynomial_ring = coefficient_ring<
        polynomial<rational_field>, typename rational_field::scalar_type>;
#endif

}// namespace lal

#endif// LIBALGEBRA_LITE_POLYNOMIAL_H
