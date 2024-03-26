//
// Created by sam on 3/25/24.
//

#ifndef POLY_RATIONAL_H
#define POLY_RATIONAL_H

#include <roughpy/core/container/unorderd_map.h>
#include <roughpy/core/hash.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/alloc.h>
#include <roughpy/platform/errors.h>

#include "types/monomial.h"

#include "arbitrary_precision_rational.h"
#include "rational.h"

namespace rpy {
namespace scalars {

template <typename Sca>
class ROUGHPY_SCALARS_NO_EXPORT Polynomial : public platform::SmallObjectBase
{
    using map_type = containers::FlatHashMap<Monomial, Sca, Hash<Monomial>>;
    using scalar_type = Sca;

    map_type m_data;

    template <typename Rng, typename = enable_if_t<ranges::range<Rng>>>
    explicit Polynomial(Rng&& rng)
        : m_data(ranges::begin(rng), ranges::end(rng))
    {}

public:
    explicit Polynomial(Monomial mon);
    Polynomial(Monomial mon, Scalar scalar);

private:
    template <typename F>
    ROUGHPY_SCALARS_NO_EXPORT void inplace_transform(F&& func);

public:
    Polynomial& operator*=(const Sca& rat)
    {
        using elt_t = typename map_type::value_type;
        inplace_transform([&rat](elt_t& elt) { elt.second *= rat; });
        return *this;
    }
    Polynomial& operator/=(const Sca& other)
    {
        using elt_t = typename map_type::value_type;
        inplace_transform([&rat](elt_t& elt) { elt.second /= rat; });
        return *this;
    }

    Polynomial& operator+=(const Polynomial& other);
    Polynomial& operator-=(const Polynomial& other);
    Polynomial& operator*=(const Polynomial& other);

private:
    template <typename F>
    ROUGHPY_SCALARS_NO_EXPORT RPY_NO_DISCARD Polynomial map_for_each(F&& func
    ) const;

public:
    friend Polynomial operator-(const Polynomial& pol)
    {
        using elt_t = typename map_type::value_type;
        return pol.map_for_each([](const elt_t& elt) -> elt_t {
            return {elt.first, -elt.second};
        });
    }

    friend Polynomial operator+(const Polynomial& lhs, const Polynomial& rhs)
    {
        Polynomial result(lhs);
        result += rhs;
        return result;
    }

    friend Polynomial operator-(const Polynomial& lhs, const Polynomial& rhs)
    {
        Polynomial result(lhs);
        result -= rhs;
        return result;
    }

    friend Polynomial operator*(const Sca& lhs, const Polynomial& rhs)
    {
        using elt_t = typename map_type::value_type;
        return rhs.map_for_each([&lhs](const elt_t& elt) -> elt_t {
            return {elt.first, lhs * elt.second};
        });
    }
    friend Polynomial operator*(const Polynomial& lhs, const Sca& rhs)
    {
        using elt_t = typename map_type::value_type;
        return lhs.map_for_each([&rhs](const elt_t& elt) -> elt_t {
            return {elt.first, elt.second * rhs};
        });
    };
    friend Polynomial operator*(const Polynomial& lhs, const Polynomial& rhs)
    {
        Polynomial result(lhs);
        result *= rhs;
        return result;
    }

    friend Polynomial operator/(const Polynomial& lhs, const Sca& rhs)
    {
        RPY_CHECK(rhs != 0);
        using elt_t = typename map_type::value_type;
        return lhs.map_for_each([&rhs](const elt_t& elt) -> elt_t {
            return {elt.first, elt.second / rhs};
        });
    }
};

template <typename Sca>
template <typename F>
void Polynomial<Sca>::inplace_transform(F&& func)
{
    m_data | views::transform(std::forward<F>(func));
}

template <typename Sca>
template <typename F>
Polynomial<Sca> Polynomial<Sca>::map_for_each(F&& func) const
{
    using elt_t = typename map_type::value_type;
    return Polynomial(
            m_data | views::transform(std::forward<F>(func))
            | views::filter([](elt_t& val) { return val != 0; }) | ranges::move
    );
}

extern template class ROUGHPY_SCALARS_NO_EXPORT
        Polynomial<ArbitraryPrecisionRational>;
extern template class ROUGHPY_SCALARS_NO_EXPORT Polynomial<Rational64>;
extern template class ROUGHPY_SCALARS_NO_EXPORT Polynomial<Rational32>;

using APPolyRat = Polynomial<ArbitraryPrecisionRational>;
using PolyRat64 = Polynomial<Rational64>;
using PolyRat32 = Polynomial<Rational32>;

}// namespace scalars
}// namespace rpy

#endif// POLY_RATIONAL_H
