//
// Created by sam on 1/29/24.
//

#include "vector.h"
#include "roughpy/core/container/vector.h"

#include "basis.h"
#include "basis_key.h"
#include "key_algorithms.h"
#include "mutable_vector_element.h"
#include "vector_iterator.h"

#include "kernels/addition_kernel.h"
#include "kernels/fused_add_left_scalar_multiply.h"
#include "kernels/fused_add_right_scalar_multiply.h"
#include "kernels/fused_sub_right_scalar_multiply_kernel.h"
#include "kernels/inplace_addition_kernel.h"
#include "kernels/inplace_left_scalar_multiply_kernel.h"
#include "kernels/inplace_right_scalar_multiply_kernel.h"
#include "kernels/inplace_subtraction_kernel.h"
#include "kernels/left_scalar_multiply_kernel.h"
#include "kernels/right_scalar_multiply_kernel.h"
#include "kernels/sparse_write_kernel.h"
#include "kernels/subtraction_kernel.h"
#include "kernels/uminus_kernel.h"

#include <roughpy/core/ranges.h>
#include <roughpy/scalars/algorithms.h>
#include <roughpy/devices/core.h>
#include <roughpy/scalars/scalar.h>

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <utility>

using namespace rpy;
using namespace algebra;

void Vector::resize_dim(rpy::dimn_t dim)
{
    const auto type = scalar_type();
    RPY_CHECK(type.is_pointer());

    // TODO: Replace this with a better implementation.

    auto new_buffer = type->allocate(dim);
    scalars::algorithms::copy(new_buffer, scalars());
    std::swap(mut_scalars(), new_buffer);

    if (is_sparse()) {
        const auto dhandle = device();
        auto new_key_buffer
                = KeyArray(dhandle->alloc(basis_key_type_info, dim));
        /*
         * I'm about to do something rather dangerous. I'm going to memcopy the
         * raw bytes of the keys over to the new buffer and then quietly release
         * the memory associated with the original key_buffer. This will have
         * the effect of clearing the memory without calling the individual
         * constructors. This is essentially using a move operation on each of
         * the keys individually.
         */
        algorithms::copy(new_key_buffer, keys());
        dhandle->raw_free(mut_keys().mut_buffer());
        mut_keys() = std::move(new_key_buffer);
    }
}

dimn_t Vector::dimension() const noexcept
{
    RPY_DBG_ASSERT(p_data != nullptr);
    return p_data->size();
}

dimn_t Vector::size() const noexcept
{
    RPY_DBG_ASSERT(p_data != nullptr);
    return p_data->size();
}

bool Vector::is_zero() const noexcept
{
    if (fast_is_zero()) { return true; }

    // TODO: This won't work if the scalars are not ordered
    const auto max = scalars::algorithms::max(p_data->scalars());
    const auto min = scalars::algorithms::min(p_data->scalars());

    return max.is_zero() && min.is_zero();
}

void Vector::set_zero()
{
    if (is_sparse()) {
        mut_scalars() = scalars::ScalarArray(scalar_type());
        mut_keys() = KeyArray();
    } else {
        scalars::algorithms::fill(mut_scalars(), 0);
    }
}

optional<dimn_t> Vector::get_index(const BasisKey& key) const noexcept
{

    if (is_dense() && key.is_index()) {
        auto index = key.get_index();
        return index >= dimension() ? optional<dimn_t>() : index;
    }

    if (!p_basis->has_key(key)) { return {}; }

    const auto* basis = p_basis.get();
    if (p_basis->is_ordered()) {
        if (is_dense()) {
            auto index = p_basis->to_index(key);
            return index >= dimension() ? optional<dimn_t>() : index;
        }

        auto keys = p_data->keys().as_slice();

        auto compare = [basis](const BasisKey& lhs, const BasisKey& rhs) {
            return basis->less(lhs, rhs);
        };

        const auto* begin = keys.begin();
        const auto* end = keys.end();
        const auto* found = std::lower_bound(begin, end, key, compare);
        if (found != end && p_basis->equals(key, *found)) {
            return static_cast<dimn_t>(found - begin);
        }
    } else {
        auto keys = p_data->keys().as_slice();
        const auto* begin = keys.begin();
        const auto* end = keys.end();

        auto compare = [basis, key](const BasisKey& arg) {
            return basis->equals(arg, key);
        };
        const auto* found = std::find_if(begin, end, compare);

        if (found != end) { return static_cast<dimn_t>(found - begin); }
    }

    return {};
}

void Vector::make_dense()
{
    if (is_dense()) { return; }
    RPY_CHECK(p_basis->is_ordered());

    const auto scalar_device = p_data->scalar_buffer().device();
    const auto key_device = p_data->key_buffer().device();

    auto dense_data = VectorDataPtr(new VectorData(scalar_type()));

    dimn_t dimension;
    if (key_device->is_host()) {
        /*
         * If we're on host, then the keys might not be an array of indices, so
         * we need to do the transformation into indices first, then look for
         * the maximum value.
         */
        dimension = 0;
        dimn_t index;
        for (auto& key : p_data->mut_keys().as_mut_slice()) {
            index = p_basis->to_index(key);
            if (index > dimension) { dimension = index; }
            key = BasisKey(index);
        }

    } else {
        /*
         * On device, all of the key should be indices so we can just use the
         * algorithms max to find the maximum index that exists in the vector.
         */
        dimension = algorithms::max(p_data->keys());
    }

    auto resize_dim = p_basis->dense_dimension(dimension);

    dense_data->resize(resize_dim);

    /*
     * We have already made sure that the buffer only contains indices so we can
     * call a kernel to write the data into the dense array. The only remaining
     * tricky part is to make sure the keys live on the same device as the
     * scalars. We can fix this by copying the data to the device if necessary.
     */

    devices::Buffer key_buffer;
    if (scalar_device == key_device) {
        key_buffer = p_data->key_buffer();
    } else {
        p_data->mut_key_buffer().to_device(key_buffer, scalar_device);
    }

    const SparseWriteKernel kernel(&*p_basis);

    kernel(*dense_data, *p_data);

    std::swap(dense_data, p_data);
}

void Vector::make_sparse()
{
    if (is_sparse()) { return; }

    KeyArray keys(dimension());
    {
        auto key_slice = keys.as_mut_slice();
        for (auto [i, k] : views::enumerate(key_slice)) { k = i; }
    }

    p_data->mut_keys() = std::move(keys);
}

Vector::Vector() = default;

Vector::~Vector() = default;

Vector::Vector(const Vector& other)
    : p_data(new VectorData(*other.p_data)),
      p_basis(other.p_basis)
{}

Vector::Vector(Vector&& other) noexcept
    : p_data(std::move(other.p_data)),
      p_basis(std::move(other.p_basis))
{}

Vector& Vector::operator=(const Vector& other)
{
    if (&other != this) {
        *p_data = VectorData(*other.p_data);
        p_basis = other.p_basis;
    }
    return *this;
}

Vector& Vector::operator=(Vector&& other) noexcept
{
    if (&other != this) {
        p_basis = std::move(other.p_basis);
        p_data = std::move(other.p_data);
    }
    return *this;
}

scalars::Scalar Vector::get(const BasisKey& key) const
{
    RPY_CHECK(p_basis->has_key(key));
    if (const auto index = get_index(key)) { return p_data->scalars()[*index]; }
    return scalars::Scalar(p_data->scalar_type());
}

scalars::Scalar Vector::get_mut(const BasisKey& key)
{
    using scalars::Scalar;
    RPY_CHECK(p_basis->has_key(key));
    return Scalar(std::make_unique<MutableVectorElement>(this, key));
}

Vector::const_iterator Vector::begin() const noexcept
{
    return {p_data->scalars().view(), p_data->keys().view(), 0};
}

Vector::const_iterator Vector::end() const noexcept
{
    return {p_data->scalars().view(), p_data->keys().view(), size()};
}

namespace {

/*
 * Note we never need to check compatibility with result vector in a true
 * binary operation since it inherits its basis and scalar type from the
 * left-hand vector input.
 */
/**
 * @brief Check that two vectors have compatible bases and scalar rings.
 * @param lhs left hand input vector/inplace result
 * @param rhs right hand input vector
 */
void check_vector_compatibility(const Vector& lhs, const Vector& rhs)
{
    RPY_CHECK(lhs.basis() == rhs.basis());
    RPY_CHECK(lhs.scalar_type() == rhs.scalar_type());
}

dimn_t compute_new_size(const Vector& lhs, const Vector& rhs) noexcept
{
    if (lhs.is_sparse() || rhs.is_sparse()) {
        return std::min(
                lhs.basis()->max_dimension(),
                lhs.dimension() + rhs.dimension()
        );
    }
    return std::max(lhs.dimension(), rhs.dimension());
}

}// namespace

void Vector::check_and_resize_for_operands(const Vector& lhs, const Vector& rhs)
{
    if (&lhs == this) {
        check_vector_compatibility(*this, rhs);
        resize_dim(compute_new_size(*this, rhs));
    } else {
        check_vector_compatibility(lhs, rhs);
        resize_dim(std::max(dimension(), compute_new_size(lhs, rhs)));
    }
}

Vector Vector::uminus() const
{
    Vector result(p_basis, scalar_type());
    result.p_data = VectorDataPtr(new VectorData(scalar_type(), dimension()));

    const UminusKernel kernel(&*p_basis);
    kernel(*result.p_data, *p_data);
    return result;
}

Vector Vector::add(const Vector& other) const
{
    Vector result(p_basis, scalar_type());
    const AdditionKernel kernel(&*p_basis);
    kernel(*result.p_data, *p_data, *other.p_data);
    return result;
}

Vector Vector::sub(const Vector& other) const
{
    Vector result(p_basis, scalar_type());
    const SubtractionKernel kernel(&*p_basis);
    kernel(*result.p_data, *p_data, *other.p_data);
    return result;
}

Vector Vector::left_smul(const scalars::Scalar& other) const
{

    Vector result(p_basis, scalar_type());
    if (!other.is_zero()) {
        const LeftScalarMultiplyKernel kernel(&*p_basis);
        kernel(*result.p_data, *p_data, other);
    }
    return result;
}

Vector Vector::right_smul(const scalars::Scalar& other) const
{
    Vector result(p_basis, scalar_type());

    if (!other.is_zero()) {
        const RightScalarMultiplyKernel kernel(&*p_basis);
        kernel(*result.p_data, *p_data, other);
    }

    return result;
}

Vector Vector::sdiv(const scalars::Scalar& other) const
{
    return right_smul(other.reciprocal());
}

Vector& Vector::add_inplace(const Vector& other)
{
    if (&other != this) {
        const InplaceAdditionKernel kernel(&*p_basis);
        kernel(*p_data, *other.p_data);
    } else {
        // Adding a vector to itself has the effect of multiplying all
        // entries by 2
        this->smul_inplace(scalars::Scalar(2));
    }
    return *this;
}

Vector& Vector::sub_inplace(const Vector& other)
{
    if (&other != this) {
        const InplaceSubtractionKernel kernel(&*p_basis);
        kernel(*p_data, *other.p_data);
    } else {
        // Subtracting a vector from itself yields zero
        this->set_zero();
    }

    return *this;
}

Vector& Vector::smul_inplace(const scalars::Scalar& other)
{
    const InplaceRightScalarMultiplyKernel kernel(&*p_basis);
    kernel(*p_data, other);
    return *this;
}

Vector& Vector::sdiv_inplace(const scalars::Scalar& other)
{
    return smul_inplace(other.reciprocal());
}

Vector& Vector::add_scal_mul(const Vector& other, const scalars::Scalar& scalar)
{
    if (&other != this) {
        const FusedAddRightScalarMultiply kernel(&*p_basis);
        kernel(*p_data, *other.p_data, scalar);
    } else {
        scalars::Scalar tmp(scalar.type_info(), 1, 1);
        tmp += scalar;
        this->smul_inplace(tmp);
    }
    return *this;
}

Vector& Vector::sub_scal_mul(const Vector& other, const scalars::Scalar& scalar)
{
    if (&other != this) {
        const FusedSubRightScalarMultiplyKernel kernel(&*p_basis);
        kernel(*p_data, *other.p_data, scalar);
    } else {
        scalars::Scalar tmp(scalar.type_info(), 1, 1);
        tmp -= scalar;
        this->smul_inplace(tmp);
    }
    return *this;
}

Vector& Vector::add_scal_div(const Vector& other, const scalars::Scalar& scalar)
{
    if (&other != this) {
        const FusedAddRightScalarMultiply kernel(&*p_basis);
        kernel(*p_data, *other.p_data, scalar.reciprocal());
    } else {
        scalars::Scalar tmp(scalar.type_info(), 1, 1);
        tmp += scalar.reciprocal();
        this->smul_inplace(tmp);
    }
    return *this;
}

Vector& Vector::sub_scal_div(const Vector& other, const scalars::Scalar& scalar)
{
    if (&other != this) {
        const FusedSubRightScalarMultiplyKernel kernel(&*p_basis);
        kernel(*p_data, *other.p_data, scalar.reciprocal());
    } else {
        scalars::Scalar tmp(scalar.type_info(), 1, 1);
        tmp -= scalar.reciprocal();
        this->smul_inplace(tmp);
    }
    return *this;
}

bool Vector::operator==(const Vector& other) const
{
    if (&other == this) { return true; }

    if (p_basis != other.p_basis || scalar_type() != other.scalar_type()) {
        return false;
    }

    if (is_dense() && other.is_dense()) {
        const auto mismatch = scalars::algorithms::mismatch(
                scalars()[{0, p_data->size()}],
                other.scalars()[{0, other.p_data->size()}]
        );

        if (!mismatch) { return true; }
        if (*mismatch == p_data->size()) {
            // Check if the content of other[size, ...] is zero
            auto count = scalars::algorithms::count(
                    other.scalars()[{*mismatch, other.p_data->size()}],
                    scalars::Scalar(other.scalar_type())
            );
            return count == (other.p_data->size() - *mismatch);
        }
        if (*mismatch == other.p_data->size()) {
            // Check if the content of this[size, ...] is zero
            auto count = scalars::algorithms::count(
                    scalars()[{*mismatch, p_data->size()}],
                    scalars::Scalar(other.scalar_type())
            );
            return count == (p_data->size() - *mismatch);
        }
        return false;
    }
    // Handle dense and sparse mixtures

    return false;
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

void Vector::insert_element(const BasisKey& key, scalars::Scalar value)
{
    if (is_dense()) {
        dimn_t index = p_basis->to_index(key);
        if (index < this->dimension()) {
            p_data->mut_scalars()[index] = value;
        } else {
            p_data->resize(p_basis->dense_dimension(index));
            p_data->mut_scalars()[index] = value;
        }

    } else if (is_sparse()) {
        auto indexOpt = this->get_index(key);
        if (indexOpt.has_value()) {
            // The element already exists, simply update it
            p_data->mut_scalars()[indexOpt.value()] = value;
        } else {
            // The element doesn't exist, we need to insert it.
            // Start by resizing our containers to make room for the new element
            dimn_t newSize = this->size() + 1;
            this->resize_dim(newSize);

            // Populate the new entries with the data we've received
            p_data->mut_scalars()[newSize - 1] = value;
            p_data->mut_keys()[newSize - 1] = key;
        }
    }
}

void Vector::delete_element(const BasisKey& key, optional<dimn_t> index_hint)
{
    if (is_dense()) {
        dimn_t index = p_basis->to_index(key);
        if (index < this->dimension()) {
            // set the item to the default initialization of the scalar type
            p_data->mut_scalars()[index] = scalars::Scalar(scalar_type());
        }
    } else if (is_sparse()) {
        auto indexOpt = this->get_index(key);
        if (indexOpt.has_value()) {
            dimn_t idx = indexOpt.value();

            // Shift left from idx to the end, effectively removing the unwanted
            // value
            auto scalar_slice = p_data->mut_scalars()[{idx, p_data->size()}];
            auto key_slice = p_data->mut_keys()[{idx, p_data->size()}];
            scalars::algorithms::shift_left(scalar_slice, 1);
            algorithms::shift_left(key_slice, 1);

            p_data->set_size(p_data->size() - 1);
        }
    }
}
