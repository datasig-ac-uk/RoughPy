//
// Created by sam on 1/29/24.
//

#include "vector.h"
#include "roughpy/core/container/vector.h"

#include "basis.h"
#include "basis_key.h"
#include "key_algorithms.h"

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

#define RPY_CONTEXT_CHECK()                                                    \
    do {                                                                       \
        if (RPY_UNLIKELY(!p_context)) {                                        \
            RPY_THROW(std::runtime_error, "vector has an invalid context");    \
        }                                                                      \
    } while (0)

scalars::ScalarCRef Vector::get(const BasisKey& key) const
{
    RPY_CONTEXT_CHECK();
    if (auto index = p_context->get_index(*this, key)) {
        return this->ScalarVector::get(*index);
    }
    RPY_THROW(std::runtime_error, "Invalid BasisKey");
}

scalars::ScalarRef Vector::get_mut(const BasisKey& key)
{
    RPY_CONTEXT_CHECK();
    if (auto index = p_context->get_index(*this, key)) {
        return this->ScalarVector::get_mut(*index);
    }
    RPY_THROW(std::runtime_error, "Invalid BasisKey");
}

VectorIterator Vector::begin() const
{
    RPY_CONTEXT_CHECK();
    return p_context->begin_iterator(this->ScalarVector::begin());
}

VectorIterator Vector::end() const
{
    RPY_CONTEXT_CHECK();
    return p_context->end_iterator(this->ScalarVector::end());
}

void Vector::check_and_resize_for_operands(const Vector& lhs, const Vector& rhs)
{
    if (RPY_UNLIKELY(lhs.p_context == nullptr && rhs.p_context == nullptr)) {
        RPY_THROW(std::runtime_error, "Invalid vector arguments");
    }
    const auto& context = lhs.p_context ? lhs.p_context : rhs.p_context;

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

    if (!p_context) { p_context = context->empty_like(); }

    this->resize_base_dim(new_base_size);
    if (new_fibre_size > 0) { this->resize_fibre_dim(new_fibre_size); }
}

void Vector::set_zero()
{
    if (RPY_LIKELY(p_context)) {
        // TODO: handle sparsity
        this->ScalarVector::set_zero();
    }
}

void Vector::insert_element(const BasisKey& key, scalars::Scalar value)
{
    RPY_CONTEXT_CHECK();
}

void Vector::delete_element(const BasisKey& key) { RPY_CONTEXT_CHECK(); }

Vector Vector::uminus() const
{
    RPY_CONTEXT_CHECK();
    UMinusOperator uminus;
    Vector result(
            p_context->copy(),
            scalar_type(),
            p_context->dimension(*this)
    );
    result.p_context
            ->unary(uminus, result, *this, scalars::ops::UnaryMinusOperator());
    return result;
}

Vector Vector::add(const Vector& other) const
{
    RPY_CONTEXT_CHECK();
    Vector result(p_context->empty_like(), scalar_type());
    AdditionOperator add;

    result.p_context->binary(
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
    RPY_CONTEXT_CHECK();
    Vector result(p_context->empty_like(), scalar_type());
    SubtractionOperator sub;

    result.p_context->binary(
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
    RPY_CONTEXT_CHECK();
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
    RPY_CONTEXT_CHECK();
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
    RPY_CONTEXT_CHECK();
    Vector result(
            p_context->copy(),
            scalar_type(),
            p_context->dimension(*this)
    );

    LeftSMulOperator left_smul;

    result.p_context->unary(
            left_smul,
            result,
            *this,
            scalars::ops::LeftMultiplyOperator(other)
    );

    return result;
}

Vector Vector::right_smul(const scalars::Scalar& other) const
{
    RPY_CONTEXT_CHECK();
    Vector result(
            p_context->copy(),
            scalar_type(),
            p_context->dimension(*this)
    );

    RightSMulOperator right_smul;

    result.p_context->unary(
            right_smul,
            result,
            *this,
            scalars::ops::RightMultiplyOperator(other)
    );

    return result;
}

Vector& Vector::smul_inplace(const scalars::Scalar& other)
{
    RPY_CONTEXT_CHECK();
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
    RPY_CONTEXT_CHECK();
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
    RPY_CONTEXT_CHECK();
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
    for (auto&& item : value) {
        os << ' ' << item->second << '(' << basis->to_string(item->first)
           << ')';
    }
    return os << " }";
}
