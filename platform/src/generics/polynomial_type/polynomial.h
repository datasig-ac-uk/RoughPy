//
// Created by sam on 26/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_H
#define ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_H

#include <iosfwd>
#include <utility>

#include <boost/container/flat_map.hpp>
#include <gmp.h>

#include "indeterminate.h"
#include "monomial.h"

namespace rpy {
namespace generics {

namespace dtl {

struct RationalCoeff {
    mpq_t content;

    RationalCoeff() { mpq_init(content); }

    // Copy constructor
    RationalCoeff(const RationalCoeff& other)
    {
        mpq_init(content);
        mpq_set(content, other.content);
    }

    // Move constructor
    RationalCoeff(RationalCoeff&& other) noexcept
    {
        mpq_init(content);
        mpq_swap(content, other.content);
    }

    RationalCoeff(int64_t num, int64_t den)
    {
        mpq_init(content);
        mpq_set_si(content, num, den);
    }

    // Copy assignment operator
    RationalCoeff& operator=(const RationalCoeff& other)
    {
        if (this != &other) { mpq_set(content, other.content); }
        return *this;
    }

    // Move assignment operator
    RationalCoeff& operator=(RationalCoeff&& other) noexcept
    {
        if (this != &other) {
            mpq_swap(content, other.content);
            mpq_set_si(other.content, 0, 1);
        }
        return *this;
    }

    ~RationalCoeff() { mpq_clear(content); }
};

}// namespace dtl

class Polynomial
    : private boost::container::flat_map<Monomial, dtl::RationalCoeff>
{
public:
    using typename flat_map::const_iterator;
    using typename flat_map::const_pointer;
    using typename flat_map::const_reference;
    using typename flat_map::difference_type;
    using typename flat_map::iterator;
    using typename flat_map::key_type;
    using typename flat_map::mapped_type;
    using typename flat_map::pointer;
    using typename flat_map::reference;
    using typename flat_map::size_type;
    using typename flat_map::value_type;

    using flat_map::flat_map;

    Polynomial(const Polynomial& other) = default;
    Polynomial(Polynomial&& other) noexcept = default;

    explicit Polynomial(dtl::RationalCoeff coeff)
        : flat_map({std::make_pair(Monomial(), std::move(coeff))})
    {}

    explicit Polynomial(const Monomial& monomial, dtl::RationalCoeff coeff)
        : flat_map({std::make_pair(monomial, std::move(coeff))})
    {}

    explicit Polynomial(
            std::initializer_list<pair<Monomial, dtl::RationalCoeff>> list
    )
        : flat_map(list)
    {}

    Polynomial& operator=(const Polynomial& other) = default;
    Polynomial& operator=(Polynomial&& other) noexcept = default;

    using flat_map::begin;
    using flat_map::cbegin;
    using flat_map::cend;
    using flat_map::crbegin;
    using flat_map::crend;
    using flat_map::end;
    using flat_map::erase;
    using flat_map::find;
    using flat_map::rbegin;
    using flat_map::rend;

    using flat_map::emplace;
    using flat_map::empty;
    using flat_map::size;
    using flat_map::clear;
    using flat_map::operator[];

    deg_t degree() const noexcept;

    bool is_constant() const noexcept;
};

void poly_add_inplace(Polynomial& lhs, const Polynomial& rhs);
void poly_sub_inplace(Polynomial& lhs, const Polynomial& rhs);
void poly_mul_inplace(Polynomial& lhs, const Polynomial& rhs);
void poly_div_inplace(Polynomial& lhs, mpq_srcptr rhs);
inline void poly_div_inplace(Polynomial& lhs, const dtl::RationalCoeff& rhs)
{
    poly_div_inplace(lhs, rhs.content);
}

bool poly_cmp_equal(const Polynomial& lhs, const Polynomial& rhs) noexcept;
inline bool poly_cmp_is_zero(const Polynomial& value) { return value.empty(); }

RPY_NO_DISCARD hash_t hash_value(const Polynomial& value);

void poly_print(std::ostream& os, const Polynomial& value);


inline bool operator==(const Polynomial& lhs, const Polynomial& rhs) noexcept
{
    return poly_cmp_equal(lhs, rhs);
}

inline std::ostream& operator<<(std::ostream& os, const Polynomial& value)
{
    poly_print(os, value);
    return os;
}

}// namespace generics

}// namespace rpy

#endif// ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_H
