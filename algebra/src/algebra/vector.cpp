//
// Created by sam on 1/29/24.
//

#include "vector.h"
#include "roughpy/core/container/vector.h"

#include "basis.h"
#include "key_algorithms.h"

#include <roughpy/core/ranges.h>
#include <roughpy/device_support/operators.h>
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

#define RPY_CONTEXT_CHECK()                                                    \
    do {                                                                       \
        if (RPY_UNLIKELY(!p_data)) {                                           \
            RPY_THROW(std::runtime_error, "vector has an invalid context");    \
        }                                                                      \
    } while (0)

void Vector::check_and_resize_for_operands(const Vector& lhs, const Vector& rhs)
{
    if (RPY_UNLIKELY(lhs.p_data == nullptr && rhs.p_data == nullptr)) {
        RPY_THROW(std::runtime_error, "Invalid vector arguments");
    }
    const auto& context = lhs.p_data ? lhs.p_data : rhs.p_data;

    dimn_t new_base_size = 0;
    dimn_t new_fibre_size = 0;

    {
        // TODO: handle sparsity
        new_base_size = std::max(lhs.size(), rhs.size());
        new_fibre_size
                = std::max(lhs.fibre_data().size(), rhs.fibre_data().size());
    }

    // We need to separate the part where we interact with this because lhs
    // might alias so we gather together the operations that interact with this
    // below. We might need to put in a barrier instruction of some sort to
    // prevent the compiler from messing with this ordering.

    if (!p_data) { p_data = context->empty_like(); }

    this->resize_base_dim(new_base_size);
    if (new_fibre_size > 0) { this->resize_fibre_dim(new_fibre_size); }
}

void Vector::set_zero()
{
    if (RPY_LIKELY(p_data)) {
        // TODO: handle sparsity
    }
}
Vector Vector::uminus() const
{
    RPY_CONTEXT_CHECK();
    UMinusOperator uminus;
    Vector result(p_data->copy(), scalar_type(), p_data->dimension());
    result.p_data->unary(uminus, *p_data, scalars::ops::UnaryMinusOperator());
    return result;
}

Vector Vector::add(const Vector& other) const
{
    RPY_CONTEXT_CHECK();
    Vector result(p_data->empty_like(), scalar_type());
    AdditionOperator add;

    result.p_data->binary(
            add,
            *p_data,
            *other.p_data,
            scalars::ops::AdditionOperator()
    );
    return result;
}

Vector Vector::sub(const Vector& other) const
{
    RPY_CONTEXT_CHECK();
    Vector result(p_data->empty_like(), scalar_type());
    SubtractionOperator sub;

    result.p_data->binary(
            sub,
            *p_data,
            *other.p_data,
            scalars::ops::SubtractionOperator()
    );

    return result;
}

Vector& Vector::add_inplace(const Vector& other)
{
    RPY_CONTEXT_CHECK();
    AdditionOperator add_inplace;

    p_data->binary_inplace(
            add_inplace,
            *other.p_data,
            scalars::ops::AdditionOperator()
    );
    return *this;
}

Vector& Vector::sub_inplace(const Vector& other)
{
    RPY_CONTEXT_CHECK();
    SubtractionOperator sub_inplace;

    p_data->binary_inplace(
            sub_inplace,
            *other.p_data,
            scalars::ops::SubtractionOperator()
    );
    return *this;
}

Vector Vector::left_smul(const scalars::Scalar& other) const
{
    RPY_CONTEXT_CHECK();
    Vector result(p_data->copy(), scalar_type(), p_data->dimension());

    LeftSMulOperator left_smul;

    result.p_data->unary(
            left_smul,
            *p_data,
            scalars::ops::LeftMultiplyOperator(other)
    );

    return result;
}

Vector Vector::right_smul(const scalars::Scalar& other) const
{
    RPY_CONTEXT_CHECK();
    Vector result(p_data->copy(), scalar_type(), p_data->dimension());

    RightSMulOperator right_smul;

    result.p_data->unary(
            right_smul,
            *p_data,
            scalars::ops::RightMultiplyOperator(other)
    );

    return result;
}

Vector& Vector::smul_inplace(const scalars::Scalar& other)
{
    RPY_CONTEXT_CHECK();
    RightSMulOperator right_smul_inplace;

    p_data->unary_inplace(
            right_smul_inplace,
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
    RPY_CONTEXT_CHECK();
    ASMOperator add_scal_mul;

    p_data->binary_inplace(
            add_scal_mul,
            *other.p_data,
            scalars::ops::FusedRightMultiplyAddOperator(scalar)
    );

    return *this;
}

Vector& Vector::sub_scal_mul(const Vector& other, const scalars::Scalar& scalar)
{
    RPY_CONTEXT_CHECK();
    SSMOperator sub_scal_mul;

    p_data->binary_inplace(
            sub_scal_mul,
            *other.p_data,
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
    for (auto&& item : value) {
        os << ' ' << item->second << '(' << basis->to_string(item->first)
           << ')';
    }
    return os << " }";
}
