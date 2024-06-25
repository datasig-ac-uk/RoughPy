//
// Created by sam on 25/06/24.
//

#include "scalar_vector.h"

#include <roughpy/devices/algorithms.h>

using namespace rpy;
using namespace rpy::scalars;

void ScalarVector::resize_dim(dimn_t new_dim)
{
    const auto type = scalar_type();
    RPY_CHECK(type != nullptr);

    auto new_buffer = type->allocate(device(), new_dim);
    devices::algorithms::copy(new_buffer, p_base->scalar_buffer());
    p_base->mut_scalar_buffer() = std::move(new_buffer);
}
void ScalarVector::set_zero() const noexcept
{
    if (p_base != nullptr) {
        devices::algorithms::fill(p_base->mut_scalars(), scalar_type()->zero());
    }
    if (p_fibre != nullptr) {
        devices::algorithms::fill(
                p_fibre->mut_scalars(),
                scalar_type()->zero()
        );
    }
}
ScalarVector ScalarVector::base() const noexcept
{
    return ScalarVector(p_base, nullptr);
}
ScalarVector ScalarVector::fibre() const noexcept
{
    return ScalarVector(p_fibre, nullptr);
}
dimn_t ScalarVector::dimension() const noexcept
{
    return (p_base == nullptr) ? 0 : p_base->size();
}
dimn_t ScalarVector::size() const noexcept
{
    if (p_base == nullptr) { return 0; }

    return p_base->size()
            - devices::algorithms::count(scalars(), scalar_type()->zero());
}
bool ScalarVector::is_zero() const noexcept
{
    if (fast_is_zero()) { return true; }
    return size() == 0;
}
ScalarVector::const_reference ScalarVector::get(dimn_t index) const
{
    RPY_CHECK(device()->is_host());
    if (index < dimension()) { return p_base->scalars()[index]; }
    return scalar_type()->zero();
}
ScalarVector::reference ScalarVector::get_mut(dimn_t index)
{
    RPY_CHECK(device()->is_host());
    RPY_CHECK(index < dimension());
    return mut_scalars()[index];
}
ScalarVector::const_iterator ScalarVector::begin() const noexcept
{
    return {};
}
ScalarVector::const_iterator ScalarVector::end() const noexcept
{
    return {};
}
ScalarVector ScalarVector::uminus() const
{
    ScalarVector result(scalar_type(), dimension());
}

ScalarVector ScalarVector::add(const ScalarVector& other) const
{


    return {};
}
ScalarVector ScalarVector::sub(const ScalarVector& other) const
{
    return {};
}
ScalarVector ScalarVector::left_smul(const Scalar& scalar) const
{
    return {};
}

ScalarVector ScalarVector::right_smul(const Scalar& scalar) const
{
    return {};
}
ScalarVector ScalarVector::sdiv(const Scalar& scalar) const
{
    return {};
}
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
ScalarVector& ScalarVector::sdiv_inplace(const Scalar& other)
{

    return *this;
}
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
bool ScalarVector::operator==(const ScalarVector& other) const
{
    if (&other == this) { return true; }

    const auto mismatch
            = devices::algorithms::mismatch(scalars(), other.scalars());
    return static_cast<bool>(mismatch);
}
