//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_LIE_H
#define ROUGHPY_ALGEBRA_LIE_H

#include "algebra.h"
#include "lie_basis.h"

#include "roughpy_algebra_export.h"

#include <roughpy/platform/serialization.h>

namespace rpy {
namespace algebra {

/**
 * @class LieMultiplication
 * @brief A class representing Lie algebra multiplication operations.
 *
 * This class provides static functions for Lie algebra multiplication
 * operations, such as basis compatibility check, fused multiply-accumulate, and
 * multiplication into a vector.
 *
 */
class ROUGHPY_ALGEBRA_EXPORT LieMultiplication
    : public RcBase<LieMultiplication>
{
public:
    static bool basis_compatibility_check(const Basis& basis) noexcept;

    static void fma(Vector& out, const Vector& left, const Vector& right);
    static void multiply_into(Vector& out, const Vector& right);
};

/**
 * @class Lie
 * @brief A class representing a Lie algebra.
 *
 * This class is derived from the GradedAlgebra class and specializes in Lie
 * algebra operations. It inherits all the member variables and functions from
 * the base class.
 */
class ROUGHPY_ALGEBRA_EXPORT Lie : public GradedAlgebra<LieMultiplication>
{
    using base_t = GradedAlgebra<LieMultiplication>;

public:
    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(Lie) {}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_LIE_H
