//
// Created by sam on 1/30/24.
//

#ifndef ROUGHPY_BUNDLE_H
#define ROUGHPY_BUNDLE_H


#include "algebra_fwd.h"
#include "roughpy_algebra_export.h"

#include <roughpy/core/types.h>
#include <roughpy/core/traits.h>

#include <roughpy/scalars/scalar.h>


#include <ostream>

namespace rpy {
namespace algebra {


template <typename Base, typename Fibre>
struct BundleTraits {

    static Fibre left_action(const Base& base, const Fibre& fibre);

    static Fibre right_action(const Fibre& fibre, const Base& base);

};

template <typename Base>
struct BundleTraits<Base, Base> {
    static Base left_action(const Base& base,
                            const Base& fibre) { return base.mul(fibre); }

    static Base right_action(const Base& fibre, const Base& base)
    {
        return fibre.mul(base);
    }
};

template <typename Base, typename Fibre>
class ROUGHPY_ALGEBRA_EXPORT Bundle {
    Base m_base;
    Fibre m_fibre;

    using traits = BundleTraits<Base, Fibre>;
public:

    Bundle(Base base, Fibre fibre)
        : m_base(std::move(base)), m_fibre(std::move(fibre)) {}


    RPY_NO_DISCARD Bundle uminus() const;

    RPY_NO_DISCARD Bundle add(const Bundle& other) const;

    RPY_NO_DISCARD Bundle sub(const Bundle& other) const;

    RPY_NO_DISCARD Bundle smul(const scalars::Scalar& other) const;

    RPY_NO_DISCARD Bundle sdiv(const scalars::Scalar& other) const;

    RPY_NO_DISCARD Bundle mul(const Bundle& other) const;

    Bundle& add_inplace(const Bundle& other);

    Bundle& sub_inplace(const Bundle& other);

    Bundle& smul_inplace(const scalars::Scalar& other);

    Bundle& sdiv_inplace(const scalars::Scalar& other);

    Bundle& mul_inplace(const Bundle& other);

    Bundle& add_scal_mul(const Bundle& other, const scalars::Scalar& scalar);

    Bundle& sub_scal_mul(const Bundle& other, const scalars::Scalar& scalar);

    Bundle& add_scal_div(const Bundle& other, const scalars::Scalar& scalar);

    Bundle& sub_scal_div(const Bundle& other, const scalars::Scalar& scalar);

    Bundle& add_mul(const Bundle& lhs, const Bundle& rhs);

    Bundle& sub_mul(const Bundle& lhs, const Bundle& rhs);

    Bundle& mul_smul(const Bundle& lhs, const scalars::Scalar& rhs);

    Bundle& mul_sdiv(const Bundle& lhs, const scalars::Scalar& rhs);

    std::ostream& print(std::ostream& os) const;


    RPY_NO_DISCARD friend bool operator==(const Bundle& lhs, const Bundle& rhs)
    {
        return lhs.m_base == rhs.m_base && lhs.m_fibre == rhs.m_fibre;
    }

    RPY_NO_DISCARD friend bool operator!=(const Bundle& lhs, const Bundle& rhs)
    {
        return lhs.m_base != rhs.m_base || lhs.m_fibre != rhs.m_fibre;
    }

};


template <typename Base, typename Fibre>
std::ostream& operator<<(std::ostream& os, const Bundle<Base, Fibre>& arg)
{
    return arg.print(os);
}


template <typename Base, typename Fibre>
Bundle<Base, Fibre> Bundle<Base, Fibre>::uminus() const
{
    return {m_base.uminus(), m_fibre.uminus()};
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre> Bundle<Base, Fibre>::add(const Bundle& other) const
{
    return {m_base.add(other.m_base), m_fibre.add(other.m_fibre)};
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre> Bundle<Base, Fibre>::sub(const Bundle& other) const
{
    return {m_base.sub(other.m_base), m_fibre(other.m_fibre)};
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre> Bundle<Base,
                           Fibre>::smul(const scalars::Scalar& other) const
{
    return {m_base.smul(other), m_fibre.smul(other)};
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre> Bundle<Base,
                           Fibre>::sdiv(const scalars::Scalar& other) const
{
    return {m_base.sdiv(other), m_fibre.sdiv(other)};
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre> Bundle<Base, Fibre>::mul(const Bundle& other) const
{
    auto left = traits::left_action(m_base, other.m_fibre);
    auto right = traits::right_action(m_fibre, other.m_base);

    return {m_base.mul(other.m_base), left.add(right)};
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::add_inplace(const Bundle& other)
{
    m_base.add_inplace(other.m_base);
    m_fibre.add_inplace(other.m_fibre);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::sub_inplace(const Bundle& other)
{
    m_base.sub_inplace(other.m_base);
    m_fibre.sub_inplace(other.m_fibre);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base,
                            Fibre>::smul_inplace(const scalars::Scalar& other)
{
    m_base.smul_inplace(other);
    m_fibre.smul_inplace(other);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base,
                            Fibre>::sdiv_inplace(const scalars::Scalar& other)
{
    m_base.sdiv_inplace(other);
    m_fibre.sdiv_inplace(other);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::mul_inplace(const Bundle& other)
{
    m_base.mul_inplace(other.m_base);
    auto left = traits::left_action(m_base, other.m_fibre);
    auto right = traits::right_action(m_fibre, other.m_base);
    m_fibre = left.add(right);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::add_scal_mul(const Bundle& other,
                                                       const scalars::Scalar& scalar)
{
    m_base.add_scal_mul(other.m_base, scalar);
    m_fibre.add_scal_mul(other.m_fibre, scalar);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::sub_scal_mul(const Bundle& other,
                                                       const scalars::Scalar& scalar)
{
    m_base.sub_scal_mul(other.m_base, scalar);
    m_fibre.sub_scal_mul(other.m_fibre, scalar);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::add_scal_div(const Bundle& other,
                                                       const scalars::Scalar& scalar)
{
    m_base.add_scal_div(other.m_base, scalar);
    m_fibre.add_scal_div(other.m_fibre, scalar);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::sub_scal_div(const Bundle& other,
                                                       const scalars::Scalar& scalar)
{
    m_base.sub_scal_div(other.m_base, scalar);
    m_fibre.sub_scal_div(other.m_fibre, scalar);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::add_mul(const Bundle& lhs,
                                                  const Bundle& rhs)
{
    m_base.add_mul(lhs.m_base, rhs.m_base);
    m_fibre.add_inplace(traits::left_action(lhs.m_base, rhs.m_fibre));
    m_fibre.add_inplace(traits::right_action(lhs.m_fibre, rhs.m_base));
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::sub_mul(const Bundle& lhs,
                                                  const Bundle& rhs)
{
    m_base.sub_mul(lhs.m_base, rhs.m_base);
    m_fibre.sub_inplace(traits::left_action(lhs.m_base, rhs.m_fibre));
    m_fibre.sub_inplace(traits::right_action(lhs.m_fibre, rhs.m_base));
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::mul_smul(const Bundle& lhs,
                                                   const scalars::Scalar& rhs)
{
    m_base.mul_smul(lhs.m_base, rhs);
    m_fibre.add_scal_mul(traits::left_action(m_base, lhs.m_fibre), rhs);
    m_fibre.add_scal_mul(traits::right_action(m_fibre, lhs.m_base), rhs);
    return *this;
}

template <typename Base, typename Fibre>
Bundle<Base, Fibre>& Bundle<Base, Fibre>::mul_sdiv(const Bundle& lhs,
                                                   const scalars::Scalar& rhs)
{
    m_base.mul_sdiv(lhs.m_base, rhs);
    m_fibre.add_scal_div(traits::left_action(m_base, lhs.m_fibre), rhs);
    m_fibre.add_scal_div(traits::right_action(m_fibre, lhs.m_base), rhs);
    return *this;
}

template <typename Base, typename Fibre>
std::ostream& Bundle<Base, Fibre>::print(std::ostream& os) const
{
    os << '(' << m_base << ", " << m_fibre << ')';
    return os;
}


}
}


#endif //ROUGHPY_BUNDLE_H
