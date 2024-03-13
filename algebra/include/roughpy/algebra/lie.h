//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_LIE_H
#define ROUGHPY_ALGEBRA_LIE_H

#include "algebra.h"
#include "lie_basis.h"

#include "roughpy_algebra_export.h"

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT Lie : public Algebra
{
public:
    Lie();

    explicit Lie(Vector&& data);
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_LIE_H
