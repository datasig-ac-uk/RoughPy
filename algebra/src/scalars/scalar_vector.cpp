//
// Created by sam on 25/06/24.
//

#include "scalar_vector.h"
#include "vector_data.h"
#include <roughpy/devices/algorithms.h>

using namespace rpy;
using namespace rpy::scalars;

ScalarVector::ScalarVector() = default;
ScalarVector::ScalarVector(TypePtr scalar_type, dimn_t size)
    : m_base_data(std::move(scalar_type), size)
{}

ScalarVector::~ScalarVector() = default;

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

    auto new_buffer = type->allocate(device(), new_dim);
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
ScalarVector::const_reference ScalarVector::get(dimn_t index) const
{
    RPY_CHECK(device()->is_host());
    if (index < dimension()) { return base_data()[index]; }
    return scalar_type()->zero();
}
ScalarVector::reference ScalarVector::get_mut(dimn_t index)
{
    RPY_CHECK(device()->is_host());
    RPY_CHECK(index < dimension());
    return mut_base_data()[index];
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

    return result;
}

ScalarVector ScalarVector::add(const ScalarVector& other) const { return {}; }
ScalarVector ScalarVector::sub(const ScalarVector& other) const { return {}; }
ScalarVector ScalarVector::left_smul(const Scalar& scalar) const { return {}; }

ScalarVector ScalarVector::right_smul(const Scalar& scalar) const { return {}; }
ScalarVector ScalarVector::sdiv(const Scalar& scalar) const { return {}; }
ScalarVector& ScalarVector::add_inplace(const ScalarVector& other)
{

    return *this;
}
ScalarVector& ScalarVector::sub_inplace(const ScalarVector& other)
{

    return *this;
}
ScalarVector& ScalarVector::left_smul_inplace(const Scalar& other)
{

    return *this;
}
ScalarVector& ScalarVector::right_smul_inplace(const Scalar& other)
{
    return *this;
}
ScalarVector& ScalarVector::sdiv_inplace(const Scalar& other) { return *this; }
ScalarVector&
ScalarVector::add_scal_mul(const ScalarVector& other, const Scalar& scalar)
{
    return *this;
}
ScalarVector&
ScalarVector::sub_scal_mul(const ScalarVector& other, const Scalar& scalar)
{
    return *this;
}
ScalarVector&
ScalarVector::add_scal_div(const ScalarVector& other, const Scalar& scalar)
{
    return *this;
}
ScalarVector&
ScalarVector::sub_scal_div(const ScalarVector& other, const Scalar& scalar)
{
    return *this;
}

namespace {

bool check_all_zero(ScalarArray array)
{
    auto count = devices::algorithms::count(array, array.type()->zero());
    return count == array.size();
}

}// namespace

bool ScalarVector::operator==(const ScalarVector& other) const
{
    if (&other == this) { return true; }

    const auto mismatch
            = devices::algorithms::mismatch(base_data(), other.base_data());
    if (!mismatch) { return true; }

    const auto& index = *mismatch;

    if (index >= base_dimension()) {
        return check_all_zero(other.base_data()[{index, other.base_dimension()}]);
    }

    if (index >= other.base_dimension()) {
        return check_all_zero(base_data()[{index, base_dimension()}]);
    }

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
