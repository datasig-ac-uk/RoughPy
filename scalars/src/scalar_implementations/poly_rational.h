//
// Created by sam on 3/25/24.
//

#ifndef POLY_RATIONAL_H
#define POLY_RATIONAL_H

#include <roughpy/core/container/unordered_map.h>
#include <roughpy/core/hash.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/alloc.h>
#include <roughpy/platform/errors.h>

#include "scalar.h"

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

    map_type m_data{};

    template <typename Rng, typename = enable_if_t<ranges::range<Rng>>>
    explicit Polynomial(Rng&& rng)
        : m_data(ranges::begin(rng), ranges::end(rng))
    {}

public:
    using value_type = typename map_type::value_type;
    using reference = scalar_type&;
    using const_reference = const scalar_type&;
    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;

    Polynomial() = default;
    Polynomial(const Polynomial& other) = default;
    Polynomial(Polynomial&& other) noexcept = default;

    explicit Polynomial(Sca scalar)
        : m_data{
                  {{}, std::move(scalar)}
    }
    {}
    explicit Polynomial(Monomial mon);
    Polynomial(Monomial mon, const Scalar& scalar);

    Polynomial& operator=(const Polynomial& other) = default;
    Polynomial& operator=(Polynomial&& other) noexcept = default;

    RPY_NO_DISCARD bool empty() const noexcept { return m_data.empty(); }
    RPY_NO_DISCARD dimn_t size() const noexcept { return m_data.size(); }

    RPY_NO_DISCARD iterator begin() noexcept { return m_data.begin(); }
    RPY_NO_DISCARD iterator end() noexcept { return m_data.end(); }
    RPY_NO_DISCARD const_iterator begin() const noexcept
    {
        return m_data.begin();
    }
    RPY_NO_DISCARD const_iterator end() const noexcept { return m_data.end(); }

    RPY_NO_DISCARD const_reference operator[](const Monomial& arg) const
    {
        static const Sca zero = Sca(0);
        const auto it = m_data.find(arg);
        if (it != m_data.end()) { return it->second; }
        return zero;
    }

    RPY_NO_DISCARD reference operator[](const Monomial& arg)
    {
        return m_data[arg];
    }

private:
    template <typename F>
    ROUGHPY_SCALARS_NO_EXPORT void inplace_transform(F&& func);

    template <typename F>
    void inplace_binary(const Polynomial& rhs, F&& func);

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
        inplace_transform([&other](elt_t& elt) { elt.second /= other; });
        return *this;
    }

    Polynomial& operator+=(const Polynomial& other)
    {
        inplace_binary(other, std::plus<>());
        return *this;
    }
    Polynomial& operator-=(const Polynomial& other)
    {
        inplace_binary(other, std::minus<>());
        return *this;
    }
    Polynomial& operator*=(const Polynomial& other) { return *this; }

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
    ranges::for_each(m_data, std::forward<F>(func));
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

template <typename Sca>
inline std::ostream& operator<<(std::ostream& os, const Polynomial<Sca>& arg)
{
    os << '{';
    for (const auto& [key, val] : arg) {
        os << ' ' << val << '(' << key << ')';
    }
    os << " }";
    return os;
}

template <typename Sca>
Polynomial<Sca>::Polynomial(Monomial mon)
    : m_data{
              {mon, Sca(1)}
}
{}
template <typename Sca>
Polynomial<Sca>::Polynomial(Monomial mon, const Scalar& scalar)
    : m_data{
              {mon, scalar_cast<Sca>(scalar)}
}
{}

template <typename Sca>
template <typename F>
void Polynomial<Sca>::inplace_binary(const Polynomial& rhs, F&& func)
{}

template <typename Sca>
bool operator==(const Polynomial<Sca>& lhs, const Polynomial<Sca>& rhs) noexcept
{
    return ranges::equal(lhs, rhs);
}

template <typename Sca>
bool operator!=(const Polynomial<Sca>& lhs, const Polynomial<Sca>& rhs) noexcept
{
    return !ranges::equal(lhs, rhs);
}

extern template class ROUGHPY_SCALARS_NO_EXPORT
        Polynomial<ArbitraryPrecisionRational>;
extern template class ROUGHPY_SCALARS_NO_EXPORT Polynomial<Rational64>;
extern template class ROUGHPY_SCALARS_NO_EXPORT Polynomial<Rational32>;

using APPolyRat = Polynomial<ArbitraryPrecisionRational>;
using PolyRat64 = Polynomial<Rational64>;
using PolyRat32 = Polynomial<Rational32>;

}// namespace scalars

namespace devices {
namespace dtl {

template <>
struct type_code_of_impl<scalars::APPolyRat> {
    static constexpr TypeCode value = TypeCode::APRationalPolynomial;
};

template <typename T>
struct type_code_of_impl<scalars::Polynomial<scalars::Rational<T>>> {
    static constexpr TypeCode value = TypeCode::Polynomial;
};

template <typename T>
struct type_size_of_impl<scalars::Polynomial<scalars::Rational<T>>> {
    static constexpr dimn_t value = sizeof(T);
};

}// namespace dtl
}// namespace devices

}// namespace rpy

#endif// POLY_RATIONAL_H
