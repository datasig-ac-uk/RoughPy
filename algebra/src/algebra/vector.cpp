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
#include <roughpy/scalars/scalars_fwd.h>

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

Vector::~Vector() = default;

void Vector::check_and_resize_for_operands(const Vector& lhs, const Vector& rhs)
{
    dimn_t new_base_size = 0;
    dimn_t new_fibre_size = 0;

    // {
    // TODO: handle sparsity
    // new_base_size = std::max(lhs.size(), rhs.size());
    // new_fibre_size
    // = std::max(lhs.fibre_data().size(), rhs.fibre_data().size());
    // }

    // We need to separate the part where we interact with this because lhs
    // might alias so we gather together the operations that interact with this
    // below. We might need to put in a barrier instruction of some sort to
    // prevent the compiler from messing with this ordering.

    // if (!p_data) { p_data = context->empty_like(); }

    // this->resize_base_dim(new_base_size);
    // if (new_fibre_size > 0) { this->resize_fibre_dim(new_fibre_size); }
}

void Vector::set_zero()
{
    scalars::algorithms::fill(mut_base_data(), scalar_type()->zero());
}

void Vector::insert_element(BasisKeyCRef key, scalars::Scalar value)
{
    const auto index = p_basis->to_index(std::move(key));
    if (RPY_UNLIKELY(index < dimension())) {
        resize_base_dim(p_basis->dense_dimension(index+1));
    }

    ScalarVector::base_get_mut(index) = std::move(value);
}

void Vector::delete_element(BasisKeyCRef key)
{
    if (const auto index = get_index(std::move(key))) {
        ScalarVector::base_get_mut(*index) = scalar_type()->zero();
    }
}


Vector Vector::minus() const
{
    const UMinusOperator uminus;
    Vector result(
            ScalarVector(scalar_type(), dimension()),
            BasisPointer(p_basis)
    );
    uminus.eval(result, *this, scalars::ops::UnaryMinusOperator());
    return result;
}

Vector Vector::left_smul(ScalarCRef other) const
{
    const LeftSMulOperator left_smul;
    Vector result(
            ScalarVector(scalar_type(), dimension()),
            BasisPointer(p_basis)
    );
    left_smul.eval(result, *this, scalars::ops::LeftMultiplyOperator(other));
    return result;
}

Vector Vector::right_smul(ScalarCRef other) const
{
    const RightSMulOperator right_smul;
    Vector result(
            ScalarVector(scalar_type(), dimension()),
            BasisPointer(p_basis)
    );
    right_smul.eval(result, *this, scalars::ops::RightMultiplyOperator(other));
    return result;
}

Vector Vector::add(const Vector& rhs) const
{
    Vector result;
    if (rhs.is_dense()) {
        result = Vector(ScalarVector::add(rhs), BasisPointer(p_basis));
    } else {
        result = *this;
        result += rhs;
    }

    return result;
}

Vector Vector::sub(const Vector& rhs) const
{
    Vector result;
    if (rhs.is_dense()) {
        result = Vector(ScalarVector::sub(rhs), BasisPointer(p_basis));
    } else {
        result = *this;
        result -= rhs;
    }

    return result;
}

namespace {

template <typename F>
void fallback_inplace_binary(Vector& lhs, const Vector& rhs, F&& op)
{
    const auto& basis = *lhs.basis();
    {
        const auto end = rhs.base_end();
        for (auto it = rhs.base_begin(); it != end; ++it) {
            const auto kv = *it;
            auto index = basis.to_index(kv.first);
            if (RPY_UNLIKELY(index > lhs.dimension())) {
                lhs.ScalarVector::resize_base_dim(index + 1);
            }
            op(lhs.ScalarVector::base_get_mut(index), kv.second);
        }
    }

    {
        const auto end = rhs.fibre_end();
        for (auto it = rhs.fibre_begin(); it != end; ++it) {
            const auto kv = *it;
            auto index = basis.to_index(kv.first);
            if (RPY_UNLIKELY(index > lhs.fibre_dimension())) {
                lhs.ScalarVector::resize_fibre_dim(index + 1);
            }
            op(lhs.ScalarVector::fibre_get_mut(index), kv.second);
        }
    }
}

auto get_addition(const Vector& lhs, const Vector& rhs)
{
    return [arithmetic = devices::Arithmetic(
                    lhs.scalar_type().get(),
                    rhs.scalar_type().get(),
                    devices::check_addition
            )](scalars::ScalarRef lhs, scalars::ScalarCRef rhs) {
        arithmetic.add_inplace(std::move(lhs), std::move(rhs));
    };
}

auto get_subtraction(const Vector& lhs, const Vector& rhs)
{
    return [arithmetic = devices::Arithmetic(
                    lhs.scalar_type().get(),
                    rhs.scalar_type().get(),
                    devices::check_subtraction
            )](scalars::ScalarRef lhs, scalars::ScalarCRef rhs) {
        arithmetic.sub_inplace(std::move(lhs), std::move(rhs));
    };
}

auto get_asm(const Vector& lhs, const Vector& rhs, scalars::ScalarCRef scalar)
{
    return [arithmetic = devices::Arithmetic(
                    &*lhs.scalar_type(),
                    &*rhs.scalar_type(),
                    devices::check_addition,
                    devices::check_multiplication
            ),
            mul = std::move(scalar
            )](scalars::ScalarRef lhs, scalars::ScalarCRef rhs) {
        auto prod = arithmetic.mul(rhs, mul);
        arithmetic.add_inplace(lhs, std::move(prod));
    };
}

auto get_ssm(const Vector& lhs, const Vector& rhs, scalars::ScalarCRef scalar)
{
    return [arithmetic = devices::Arithmetic(
                    &*lhs.scalar_type(),
                    &*rhs.scalar_type(),
                    devices::check_subtraction,
                    devices::check_multiplication
            ),
            mul = std::move(scalar
            )](scalars::ScalarRef lhs, scalars::ScalarCRef rhs) {
        auto prod = arithmetic.mul(rhs, mul);
        arithmetic.sub_inplace(lhs, std::move(prod));
    };
}

}// namespace

Vector& Vector::smul_inplace(ScalarCRef other)
{
    if (devices::is_zero(other)) {
        set_zero();
    } else {
        ScalarVector::right_smul_inplace(other);
    }
    return *this;
}

Vector& Vector::add_inplace(const Vector& rhs)
{
    RPY_CHECK(is_dense());
    if (rhs.is_dense()) {
        ScalarVector::add_inplace(rhs);
    } else {
        fallback_inplace_binary(*this, rhs, get_addition(*this, rhs));
    }

    return *this;
}

Vector& Vector::sub_inplace(const Vector& rhs)
{
    RPY_CHECK(is_dense());
    if (rhs.is_dense()) {
        ScalarVector::sub_inplace(rhs);
    } else {
        fallback_inplace_binary(*this, rhs, get_subtraction(*this, rhs));
    }
    return *this;
}

Vector& Vector::add_scal_mul(const Vector& other, ScalarCRef scalar)
{
    if (!devices::is_zero(scalar)) {
        RPY_CHECK(is_dense());
        if (other.is_dense()) {
            ScalarVector::add_scal_mul(other, scalar);
        } else {
            fallback_inplace_binary(
                    *this,
                    other,
                    get_asm(*this, other, std::move(scalar))
            );
        }
    }
    return *this;
}

Vector& Vector::sub_scal_mul(const Vector& other, ScalarCRef scalar)
{
    if (!devices::is_zero(scalar)) {
        RPY_CHECK(is_dense());
        if (other.is_dense()) {
            ScalarVector::sub_scal_mul(other, scalar);
        } else {
            fallback_inplace_binary(
                    *this,
                    other,
                    get_ssm(*this, other, std::move(scalar))
            );
        }
    }
    return *this;
}

bool Vector::is_equal(const Vector& other) const noexcept
{
    if (basis()->compare(other.basis()) != BasisComparison::IsSame) {
        return false;
    }

    if (is_dense() && other.is_dense()) {
        return static_cast<const ScalarVector&>(*this)
                == static_cast<const ScalarVector&>(other);
    }

    if (size() != other.size()) { return false; }

    if (other.is_dense()) {
        return ranges::all_of(other, [this](const auto& kv) {
            const auto index = this->get_index(kv.first);
            return index && ScalarVector::base_get(*index) == kv.second;
        });
    }

    return ranges::all_of(*this, [other](const auto& kv) {
        const auto index = other.get_index(kv.first);
        return index && other.ScalarVector::base_get(*index) == kv.second;
    });

}

std::ostream& algebra::operator<<(std::ostream& os, const Vector& value)
{
    const auto& basis = value.basis();
    os << '{';
    for (auto&& item : value) {
        os << ' ' << item.second << '(' << basis->to_string(item.first)
           << ')';
    }
    return os << " }";
}
