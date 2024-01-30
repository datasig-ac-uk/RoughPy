//
// Created by sam on 1/29/24.
//

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_H
#define ROUGHPY_ALGEBRA_ALGEBRA_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/core/

#include <roughpy/platform/devices/kernel.h>

#include "vector.h"

namespace rpy {
namespace algebra {


class ROUGHPY_ALGEBRA_EXPORT Multiplication {

};


class ROUGHPY_ALGEBRA_EXPORT Algebra : public Vector
{
    boost::intrusive_ptr<const Multiplication> p_multiplication;

public:

    RPY_NO_DISCARD Algebra multiply(const Vector& other) const;
    Algebra& multiply_inplace(const Vector& other);
};

template <typename A>
enable_if_t<is_base_of<Algebra, A>::value, A>
operator*(const A& lhs, const Vector& rhs)
{
    return A(lhs.multiply(rhs));
}

template <typename A>
enable_if_t<is_base_of<Algebra, A>::value, A&>
operator*=(A& lhs, const Vector& rhs)
{
    lhs.multiply_inplace(rhs);
    return lhs;
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_H
