//
// Created by sam on 1/29/24.
//

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_H
#define ROUGHPY_ALGEBRA_ALGEBRA_H

#include "roughpy_algebra_export.h"
#include "vector.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/devices/kernel.h>

#include <boost/smart_ptr/intrusive_ref_counter.hpp>

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT Multiplication
    : boost::intrusive_ref_counter<Multiplication>
{

public:

    virtual ~Multiplication();

    virtual devices::Kernel get_kernel(string_view suffix) const;


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
