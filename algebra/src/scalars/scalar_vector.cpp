//
// Created by sam on 25/06/24.
//

#include "scalar_vector.h"
#include "vector_data.h"

#include <roughpy/device_support/operators.h>
#include <roughpy/devices/algorithms.h>

using namespace rpy;
using namespace rpy::scalars;

ScalarVector::ScalarVector() = default;
ScalarVector::ScalarVector(TypePtr scalar_type, dimn_t size)
    : m_base_data(std::move(scalar_type), size)
{}


ScalarArray& ScalarVector::mut_base_data()
{
    RPY_CHECK(!m_base_data.is_const());
    return m_base_data;
}
ScalarArray& ScalarVector::mut_fibre_data()
{
    RPY_CHECK(!m_fibre_data.is_const());
    return m_fibre_data;
}

void ScalarVector::resize_base_dim(dimn_t new_dim)
{
    const auto type = scalar_type();
    RPY_CHECK(type != nullptr);

    auto new_buffer = device()->alloc(*type, new_dim);
    devices::algorithms::copy(new_buffer, base_data());
    mut_base_data() = ScalarArray(std::move(new_buffer));
}
void ScalarVector::resize_fibre_dim(dimn_t new_dim)
{
    const auto type = scalar_type();
    RPY_CHECK(type != nullptr);

    auto new_buffer = type->allocate(device(), new_dim);
    devices::algorithms::copy(new_buffer, fibre_data());
    mut_base_data() = ScalarArray(std::move(new_buffer));
}

void ScalarVector::set_base_zero()
{
    devices::algorithms::fill(mut_base_data(), scalar_type()->zero());
}
void ScalarVector::set_fibre_zero()
{
    devices::algorithms::fill(mut_fibre_data(), scalar_type()->zero());
}

dimn_t ScalarVector::size() const noexcept
{
    return base_buffer_size()
            - devices::algorithms::count(base_data(), scalar_type()->zero());
}

bool ScalarVector::is_zero() const noexcept
{
    if (fast_is_zero()) { return true; }
    return size() == 0;
}

ScalarVector::const_reference ScalarVector::base_get(dimn_t index) const
{
    RPY_CHECK(device()->is_host());
    if (index < dimension()) { return base_data()[index]; }
    return scalar_type()->zero();
}
ScalarVector::reference ScalarVector::base_get_mut(dimn_t index)
{
    RPY_CHECK(device()->is_host());
    RPY_CHECK(index < dimension());
    return mut_base_data()[index];
}
ScalarVector::const_reference ScalarVector::fibre_get(dimn_t index) const
{
    RPY_CHECK(device()->is_host());
    if (index < fibre_data().size()) { return fibre_data()[index]; }
    return scalar_type()->zero();
}
ScalarVector::reference ScalarVector::fibre_get_mut(dimn_t index)
{
    RPY_CHECK(device()->is_host());
    RPY_CHECK(index < fibre_data().size());
    return mut_fibre_data()[index];
}

ScalarVector::const_iterator ScalarVector::begin() const noexcept
{
    return {base_data(), 0};
}
ScalarVector::const_iterator ScalarVector::end() const noexcept
{
    return {base_data(), dimension()};
}
ScalarVector ScalarVector::uminus() const
{
    ScalarVector result(scalar_type(), dimension());
    const StandardUnaryVectorOperation<ops::Minus> op;
    op.eval(result, *this, ops::MinusOperator());
    return result;
}

ScalarVector ScalarVector::add(const ScalarVector& other) const
{
    const StandardBinaryVectorOperation<ops::Add> op;
    ScalarVector result(
            scalar_type(),
            std::max(dimension(), other.dimension())
    );
    op.eval(result, *this, other, ops::AdditionOperator());
    return result;
}
ScalarVector ScalarVector::sub(const ScalarVector& other) const
{
    const StandardBinaryVectorOperation<ops::Sub> op;
    ScalarVector result(
            scalar_type(),
            std::max(dimension(), other.dimension())
    );
    op.eval(result, *this, other, ops::SubtractionOperator());
    return result;
}
ScalarVector ScalarVector::left_smul(ScalarCRef scalar) const
{
    const StandardUnaryVectorOperation<ops::LeftScalarMultiply> op;
    ScalarVector result(scalar_type(), dimension());
    op.eval(result, *this, ops::LeftMultiplyOperator(std::move(scalar)));
    return result;
}

ScalarVector ScalarVector::right_smul(ScalarCRef scalar) const
{
    const StandardUnaryVectorOperation<ops::RightScalarMultiply> op;
    ScalarVector result(scalar_type(), dimension());
    op.eval(result, *this, ops::RightMultiplyOperator(std::move(scalar)));
    return result;
}
ScalarVector& ScalarVector::add_inplace(const ScalarVector& other)
{
    const StandardBinaryVectorOperation<ops::Add> op;
    op.eval_inplace(*this, other, ops::AdditionOperator());
    return *this;
}
ScalarVector& ScalarVector::sub_inplace(const ScalarVector& other)
{
    const StandardBinaryVectorOperation<ops::Sub> op;
    op.eval_inplace(*this, other, ops::SubtractionOperator());
    return *this;
}
ScalarVector& ScalarVector::left_smul_inplace(ScalarCRef other)
{
    const StandardUnaryVectorOperation<ops::LeftScalarMultiply> op;
    op.eval_inplace(*this, ops::LeftMultiplyOperator(std::move(other)));
    return *this;
}
ScalarVector& ScalarVector::right_smul_inplace(ScalarCRef other)
{
    const StandardUnaryVectorOperation<ops::LeftScalarMultiply> op;
    op.eval_inplace(*this, ops::LeftMultiplyOperator(std::move(other)));
    return *this;
}
ScalarVector&
ScalarVector::add_scal_mul(const ScalarVector& other, ScalarCRef scalar)
{
    const StandardBinaryVectorOperation<ops::FusedRightScalarMultiplyAdd> op;
    op.eval_inplace(*this, other, ops::FusedRightMultiplyAddOperator(std::move(scalar)));
    return *this;
}
ScalarVector&
ScalarVector::sub_scal_mul(const ScalarVector& other, ScalarCRef scalar)
{
    const StandardBinaryVectorOperation<ops::FusedRightScalarMultiplySub> op;
    op.eval_inplace(*this, other, ops::FusedRightMultiplySubOperator(std::move(scalar)));
    return *this;
}

namespace {

bool check_all_zero(const ScalarArray& array, dimn_t offset)
{
    const auto sliced_array = array[{offset, array.size()}];
    auto count = devices::algorithms::count(sliced_array, array.type()->zero());
    return count == sliced_array.size();
}

}// namespace

bool ScalarVector::operator==(const ScalarVector& other) const
{
    if (&other == this) { return true; }

    const auto mismatch
            = devices::algorithms::mismatch(base_data(), other.base_data());
    if (!mismatch) { return true; }

    const auto& index = *mismatch;

    if (index < other.base_dimension()) {
        return check_all_zero(other.base_data(), index);
    }

    if (index < base_dimension()) { return check_all_zero(base_data(), index); }

    return false;
}

dimn_t ScalarVector::base_size() const noexcept
{
    return base_buffer_size()
            - devices::algorithms::count(base_data(), scalar_type()->zero());
}
dimn_t ScalarVector::fibre_size() const noexcept
{
    return fibre_buffer_size()
            - devices::algorithms::count(fibre_data(), scalar_type()->zero());
}
