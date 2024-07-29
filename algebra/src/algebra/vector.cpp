//
// Created by sam on 1/29/24.
//

#include "vector.h"
#include "roughpy/core/container/vector.h"

#include "basis.h"
#include "basis_key.h"
#include "key_algorithms.h"
#include "vector_iterator.h"

#include <roughpy/core/ranges.h>
#include <roughpy/devices/algorithms.h>
#include <roughpy/devices/core.h>

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <utility>

using namespace rpy;
using namespace algebra;

using UMinusOperator
        = scalars::StandardUnaryVectorOperation<scalars::ops::Uminus>;
using AdditionOperator
        = scalars::StandardBinaryVectorOperation<scalars::ops::Add>;
using SubtractionOperator
        = scalars::StandardBinaryVectorOperation<scalars::ops::Sub>;

using LeftSMulOperator = scalars::StandardUnaryVectorOperation<
        scalars::ops::LeftScalarMultiply>;
using RightSMulOperator = scalars::StandardUnaryVectorOperation<
        scalars::ops::RightScalarMultiply>;

Vector Vector::uminus() const
{
    UMinusOperator uminus;
    Vector result(
            p_context->copy(),
            scalar_type(),
            p_context->dimension(*this)
    );
    p_context->unary(uminus, result, *this, scalars::ops::UnaryMinusOperator());
    return result;
}

Vector Vector::add(const Vector& other) const
{
    Vector result(p_context->empty_like(), scalar_type());
    AdditionOperator add;

    p_context->binary(
            add,
            result,
            *this,
            other,
            scalars::ops::AdditionOperator()
    );
    return result;
}

Vector Vector::sub(const Vector& other) const
{
    Vector result(p_context->empty_like(), scalar_type());
    SubtractionOperator sub;

    p_context->binary(
            sub,
            result,
            *this,
            other,
            scalars::ops::SubtractionOperator()
    );

    return result;
}

Vector& Vector::add_inplace(const Vector& other)
{
    AdditionOperator add_inplace;

    p_context->binary_inplace(
            add_inplace,
            *this,
            other,
            scalars::ops::AdditionOperator()
    );
    return *this;
}

Vector& Vector::sub_inplace(const Vector& other)
{
    SubtractionOperator sub_inplace;
    p_context->binary_inplace(
            sub_inplace,
            *this,
            other,
            scalars::ops::SubtractionOperator()
    );
    return *this;
}

Vector Vector::left_smul(const scalars::Scalar& other) const
{
    Vector result(
            p_context->copy(),
            scalar_type(),
            p_context->dimension(*this)
    );

    LeftSMulOperator left_smul;

    p_context->unary(
            left_smul,
            result,
            *this,
            scalars::ops::LeftMultiplyOperator(other)
    );

    return result;
}

Vector Vector::right_smul(const scalars::Scalar& other) const
{
    Vector result(
            p_context->copy(),
            scalar_type(),
            p_context->dimension(*this)
    );

    RightSMulOperator right_smul;

    p_context->unary(
            right_smul,
            result,
            *this,
            scalars::ops::RightMultiplyOperator(other)
    );

    return result;
}

Vector& Vector::smul_inplace(const scalars::Scalar& other)
{
    RightSMulOperator right_smul_inplace;

    p_context->unary_inplace(
            right_smul_inplace,
            *this,
            scalars::ops::RightMultiplyOperator(other)
    );

    return *this;
}

std::ostream& algebra::operator<<(std::ostream& os, const Vector& value)
{
    const auto basis = value.basis();
    os << '{';
    for (const auto& item : value) {
        os << ' ' << item->second << '(' << basis->to_string(item->first)
           << ')';
    }
    return os << " }";
}
