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

using ASMOperator = scalars::StandardBinaryVectorOperation<
        scalars::ops::FusedRightScalarMultiplyAdd>;
using SSMOperator = scalars::StandardBinaryVectorOperation<
        scalars::ops::FusedRightScalarMultiplySub>;

using scalars::ScalarVector;

scalars::ScalarCRef Vector::get(const BasisKey& key) const
{
    if (auto index = p_context->get_index(key)) {
        return this->ScalarVector::get(*index);
    }
    RPY_THROW(std::runtime_error, "Invalid BasisKey");
}

scalars::ScalarRef Vector::get_mut(const BasisKey& key)
{
    if (auto index = p_context->get_index(key)) {
        return this->ScalarVector::get_mut(*index);
    }
    RPY_THROW(std::runtime_error, "Invalid BasisKey");
}

typename Vector::const_iterator Vector::begin() const
{
    return p_context->make_const_iterator(this->ScalarVector::begin());
}

typename Vector::const_iterator Vector::end() const
{
    return p_context->make_const_iterator(this->ScalarVector::end());
}



void Vector::check_and_resize_for_operands(const Vector& lhs, const Vector& rhs)
{

}


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

Vector Vector::sdiv(const scalars::Scalar& other) const
{
    // This should catch division by zero or bad conversions
    auto recip = scalars::math::reciprocal(other);

    return right_smul(recip);
}

Vector& Vector::sdiv_inplace(const scalars::Scalar& other)
{
    auto recip = scalars::math::reciprocal(other);
    return smul_inplace(recip);
}

Vector& Vector::add_scal_mul(const Vector& other, const scalars::Scalar& scalar)
{
    ASMOperator add_scal_mul;

    p_context->binary_inplace(
            add_scal_mul,
            *this,
            other,
            scalars::ops::FusedRightMultiplyAddOperator(scalar)
    );

    return *this;
}

Vector& Vector::sub_scal_mul(const Vector& other, const scalars::Scalar& scalar)
{
    SSMOperator sub_scal_mul;

    p_context->binary_inplace(
            sub_scal_mul,
            *this,
            other,
            scalars::ops::FusedRightMultiplySubOperator(scalar)
    );

    return *this;
}

Vector& Vector::add_scal_div(const Vector& other, const scalars::Scalar& scalar)
{
    auto recip = scalars::math::reciprocal(scalar);
    return add_scal_mul(other, recip);
}

Vector& Vector::sub_scal_div(const Vector& other, const scalars::Scalar& scalar)
{
    auto recip = scalars::math::reciprocal(scalar);
    return sub_scal_mul(other, recip);
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
