//
// Created by sam on 1/29/24.
//

#include "vector.h"
#include "basis.h"
#include "basis_key.h"
#include "mutable_vector_element.h"
#include "vector_iterator.h"

#include <roughpy/scalars/devices/core.h>
#include <roughpy/scalars/scalar.h>

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <utility>

using namespace rpy;
using namespace algebra;

devices::Kernel Vector::get_kernel(
        OperationType type,
        string_view operation,
        string_view suffix
) const
{
    RPY_DBG_ASSERT(!operation.empty());
    const auto* stype = scalar_type();

    string op(operation);
    op += '_';
    op += stype->id();

    if (type != Unary && type != UnaryInplace) {
        op += (is_sparse() ? "_sparse" : "_dense");
    }

    if (!suffix.empty()) {
        op += '_';
        op += suffix;
    }

    auto kernel = stype->device()->get_kernel(op);
    RPY_CHECK(kernel);
    return *kernel;
}

devices::KernelLaunchParams Vector::get_kernel_launch_params() const
{
    /*
     * vector operations are one dimensional so we can safely set the work
     * parameters generically here based on the size of the vectors.
     */

    return devices::KernelLaunchParams(
            devices::Size3{m_data.scalar_buffer().size()},
            devices::Dim3{1}
    );
}

void Vector::resize_dim(rpy::dimn_t dim)
{
    const auto* type = scalar_type();

    auto new_buffer = type->allocate(dim);
    type->move_buffer(new_buffer, scalars());
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
        dhandle->memcopy(new_key_buffer.mut_buffer(), keys().buffer())
                .wait();
        dhandle->raw_free(mut_keys().mut_buffer());
        mut_keys() = std::move(new_key_buffer);
    }
}

void Vector::resize_degree(rpy::deg_t degree)
{
    auto new_size = p_basis->dimension_to_degree(degree);
    resize_dim(new_size);
}

dimn_t Vector::dimension() const noexcept { return m_data.size(); }

dimn_t Vector::size() const noexcept { return m_data.size(); }

bool Vector::is_zero() const noexcept
{
    if (fast_is_zero()) { return true; }

    auto kernel = get_kernel(Unary, "all_equal", "");
    auto params = get_kernel_launch_params();

    return true;
}

void Vector::set_zero()
{
    if (is_sparse()) {
        mut_scalars() = scalars::ScalarArray(scalar_type());
        mut_keys() = KeyArray();
    } else {
        auto kernel = get_kernel(UnaryInplace, "set_scalar", "");
        auto params = get_kernel_launch_params();

        kernel(params, mut_scalars().mut_buffer(), 0);
    }
}

optional<dimn_t> Vector::get_index(rpy::algebra::BasisKey key) const noexcept
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

        auto keys = m_data.keys().as_slice();

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
        auto keys = m_data.keys().as_slice();
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

void Vector::make_dense() {}

void Vector::make_sparse() {}

Vector::Vector() : p_basis(), m_scalar_buffer(), m_key_buffer() {}

Vector::~Vector() = default;

Vector::Vector(const Vector& other)
    : p_basis(other.p_basis),
      m_scalar_buffer(other.m_scalar_buffer),
      m_key_buffer(other.m_key_buffer)
{}

Vector::Vector(Vector&& other) noexcept
    : p_basis(std::move(other.p_basis)),
      m_scalar_buffer(std::move(other.m_scalar_buffer)),
      m_key_buffer(std::move(other.m_key_buffer))
{}

Vector& Vector::operator=(const Vector& other)
{
    if (&other != this) {
        this->~Vector();
        p_basis = other.p_basis;
        m_scalar_buffer = other.m_scalar_buffer;
        m_key_buffer = other.m_key_buffer;
    }
    return *this;
}

Vector& Vector::operator=(Vector&& other) noexcept
{
    if (&other != this) {
        this->~Vector();
        p_basis = std::move(other.p_basis);
        m_scalar_buffer = std::move(other.m_scalar_buffer);
        m_key_buffer = std::move(other.m_key_buffer);
    }
    return *this;
}

scalars::Scalar Vector::get(BasisKey key) const
{
    RPY_CHECK(p_basis->has_key(key));
    if (auto index = get_index(key)) { return m_scalar_buffer[*index]; }
    return scalars::Scalar();
}

scalars::Scalar Vector::get_mut(BasisKey key)
{
    using scalars::Scalar;
    RPY_CHECK(p_basis->has_key(key));
    return Scalar(std::make_unique<MutableVectorElement>(this, key));
}

Vector::const_iterator Vector::begin() const noexcept
{
    return {m_scalar_buffer.view(), m_key_buffer.view(), 0};
}

Vector::const_iterator Vector::end() const noexcept
{
    return {m_scalar_buffer.view(), m_key_buffer.view(), m_scalar_buffer.size()
    };
}

scalars::Scalar Vector::operator[](BasisKey key) const { return get(key); }

scalars::Scalar Vector::operator[](BasisKey key) { return get_mut(key); }

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

void Vector::apply_binary_kernel(
        string_view kernel_name,
        const Vector& lhs,
        const Vector& rhs,
        optional<scalars::Scalar> multiplier
)
{
    check_and_resize_for_operands(lhs, rhs);

    const auto lhs_sparse = lhs.is_sparse();
    const auto rhs_sparse = rhs.is_sparse();

    if (multiplier) {
        // Make sure the multiplier if given is of the correct type
        multiplier->change_type(m_scalar_buffer.type_info());
    }

    if (&lhs == this) {
        /*
         * This is the inplace mode of operation. All references to lhs are
         * unused in this branch, since we must maintain strict aliasing.
         *
         * The first thing to do is check that the vectors are compatible, and
         * resize *this to make sure it is large enough to accommodate the
         * result.
         *
         * Once this is done, we can start to think about arguments. For dense
         * vectors, the only troublesome part is whether the kernel needs an
         * additional multiplier argument. For the time being, we shall simply
         * pass the multiplier argument whenever it is provided, and let the
         * kernel check that it has the correct number of arguments etc.
         *
         * For sparse cases or mixed cases there are more problems. Since sparse
         * vectors need their key arrays along side the scalar buffers, we also
         * need to think about how these are passed into the kernel. If both are
         * sparse then this is easy. Otherwise, we're going to have to convert
         * one to be dense/sparse so that it matches. There may be reasonable
         * heuristics for deciding which. However, let's just work with the idea
         * that doing operations where one vector is sparse always causes the
         * other to be made sparse too (by the construction of a temporary key
         * array). This should take care of any actual problems for now.
         *
         */

        if (lhs_sparse && rhs_sparse) {
            // Both are sparse, os send keys and scalars

            auto kernel = get_kernel(
                    OperationType::BinaryInplace,
                    kernel_name,
                    "s"
            );
            // TODO: This won't work in general, since the final size isn't
            // known
            auto params = get_kernel_launch_params();

            if (multiplier) {
                kernel(params,
                       m_key_buffer.mut_buffer(),
                       m_scalar_buffer.mut_buffer(),
                       rhs.m_key_buffer.buffer(),
                       rhs.m_scalar_buffer.buffer(),
                       scalars::to_kernel_arg(*multiplier));
            } else {
                kernel(params,
                       m_key_buffer.mut_buffer(),
                       m_scalar_buffer.mut_buffer(),
                       rhs.m_key_buffer.buffer(),
                       rhs.m_scalar_buffer.buffer());
            }

            return;
        }

        if (rhs_sparse) {
            /*
             * This is the simple mixed case. Probably the most sensible
             * thing is to have mixed form kernel that performs the addition
             * the operation only on indices that are present in the sparse
             * data.
             */

            auto kernel = get_kernel(
                    OperationType::BinaryInplace,
                    kernel_name,
                    "s"
            );
            auto params = get_kernel_launch_params();

            if (multiplier) {
                kernel(params,
                       m_scalar_buffer.mut_buffer(),
                       rhs.m_key_buffer.buffer(),
                       rhs.m_scalar_buffer.buffer(),
                       scalars::to_kernel_arg(*multiplier));
            } else {
                kernel(params,
                       m_scalar_buffer.mut_buffer(),
                       rhs.m_key_buffer.buffer(),
                       rhs.m_scalar_buffer.buffer());
            }

            return;
        }

        if (lhs_sparse) {
            /*
             * For this mixed case, we'll promote *this to be dense too,
             * and then use the same logic as for dense-dense
             */
            make_dense();
        }

        // Both should be dense at this point
        RPY_DBG_ASSERT(is_dense() && rhs.is_dense());
        // both are dense, use the dense kernel
        auto kernel
                = get_kernel(OperationType::BinaryInplace, kernel_name, "d");
        auto params = get_kernel_launch_params();

        if (multiplier) {
            kernel(params,
                   m_scalar_buffer.mut_buffer(),
                   rhs.m_scalar_buffer.buffer(),
                   scalars::to_kernel_arg(*multiplier));
        } else {
            kernel(params,
                   m_scalar_buffer.mut_buffer(),
                   rhs.m_scalar_buffer.buffer());
        }

        return;
    }

    /*
     * Now the inplace case is dealt with we have to do it all again for the
     * out of place operations. The process is exactly the same, we just
     * have to do some checking to see what the output vector will look like
     * and then call the appropriate kernel.
     *
     * The only thing that is really different in this case is that we can't
     * reduce the cases by simply promoting lhs to dense, since it is not the
     * output vector.
     */

    // For now, always promote to dense if either is dense
    auto output_sparse = lhs_sparse && rhs_sparse;

    if (output_sparse) {
        // The resize vector might not have initialised the key array
        make_sparse();

        auto kernel = get_kernel(OperationType::Binary, kernel_name, "ss");
        auto params = get_kernel_launch_params();

        if (multiplier) {
            kernel(params,
                   m_key_buffer.mut_buffer(),
                   m_scalar_buffer.mut_buffer(),
                   lhs.m_key_buffer.buffer(),
                   lhs.m_scalar_buffer.buffer(),
                   rhs.m_key_buffer.buffer(),
                   rhs.m_scalar_buffer.buffer(),
                   scalars::to_kernel_arg(*multiplier));
        } else {
            kernel(params,
                   m_key_buffer.mut_buffer(),
                   m_scalar_buffer.mut_buffer(),
                   lhs.m_key_buffer.buffer(),
                   lhs.m_scalar_buffer.buffer(),
                   rhs.m_key_buffer.buffer(),
                   rhs.m_scalar_buffer.buffer());
        }

        return;
    }

    // The vector might be dense if it wasn't already sparse. So make dense
    make_dense();

    if (lhs_sparse) {
        auto kernel = get_kernel(OperationType::Binary, kernel_name, "sd");
        auto params = get_kernel_launch_params();

        if (multiplier) {
            kernel(params,
                   m_scalar_buffer.mut_buffer(),
                   lhs.m_key_buffer.buffer(),
                   lhs.m_scalar_buffer.buffer(),
                   rhs.m_scalar_buffer.buffer(),
                   scalars::to_kernel_arg(*multiplier));
        } else {
            kernel(params,
                   m_scalar_buffer.mut_buffer(),
                   lhs.m_key_buffer.buffer(),
                   lhs.m_scalar_buffer.buffer(),
                   rhs.m_scalar_buffer.buffer());
        }
    } else if (rhs_sparse) {
        auto kernel = get_kernel(OperationType::Binary, kernel_name, "ds");
        auto params = get_kernel_launch_params();

        if (multiplier) {
            kernel(params,
                   m_scalar_buffer.mut_buffer(),
                   lhs.m_scalar_buffer.buffer(),
                   rhs.m_key_buffer.buffer(),
                   rhs.m_scalar_buffer.buffer(),
                   scalars::to_kernel_arg(*multiplier));
        } else {
            kernel(params,
                   m_scalar_buffer.mut_buffer(),
                   lhs.m_scalar_buffer.buffer(),
                   rhs.m_key_buffer.buffer(),
                   rhs.m_scalar_buffer.buffer());
        }
    } else {
        auto kernel = get_kernel(OperationType::Binary, kernel_name, "dd");
        auto params = get_kernel_launch_params();

        if (multiplier) {
            kernel(params,
                   m_scalar_buffer.mut_buffer(),
                   lhs.m_scalar_buffer.buffer(),
                   rhs.m_scalar_buffer.buffer(),
                   scalars::to_kernel_arg(*multiplier));
        } else {
            kernel(params,
                   m_scalar_buffer.mut_buffer(),
                   lhs.m_scalar_buffer.buffer(),
                   rhs.m_scalar_buffer.buffer());
        }
    }
}

// TODO: Handle the short-circuit cases where one of the vectors is trivially
//  equal to zero.

Vector Vector::uminus() const
{
    auto kernel = get_kernel(Unary, "uminus", "");
    scalars::ScalarArray out_data(scalar_type(), dimension());
    KeyArray out_keys(m_key_buffer);

    auto launch_params = get_kernel_launch_params();

    kernel(launch_params, out_data.mut_buffer(), m_scalar_buffer.buffer());

    return {p_basis, std::move(out_data), std::move(out_keys)};
}

Vector Vector::add(const Vector& other) const
{
    Vector result(p_basis, scalar_type());
    result.apply_binary_kernel("add", *this, other);
    return result;
}

Vector Vector::sub(const Vector& other) const
{
    Vector result(p_basis, scalar_type());
    result.apply_binary_kernel("sub", *this, other);
    return result;
}

Vector Vector::left_smul(const scalars::Scalar& other) const
{
    const auto* stype = scalar_type();

    Vector result(p_basis, stype);

    if (!other.is_zero()) {
        auto kernel = get_kernel(Unary, "left_scalar_multiply", "");
        auto params = get_kernel_launch_params();

        result.m_scalar_buffer = scalars::ScalarArray(stype, dimension());
        result.m_key_buffer = m_key_buffer;

        kernel(params,
               result.m_scalar_buffer.mut_buffer(),
               m_scalar_buffer.buffer(),
               scalars::to_kernel_arg(other));
    }

    return result;
}

Vector Vector::right_smul(const scalars::Scalar& other) const
{
    const auto* stype = scalar_type();
    Vector result(p_basis, stype);

    if (!other.is_zero()) {
        auto kernel = get_kernel(Unary, "right_scalar_multiply", "");
        auto params = get_kernel_launch_params();

        result.m_scalar_buffer = scalars::ScalarArray(stype, dimension());
        result.m_key_buffer = m_key_buffer;

        kernel(params,
               result.m_scalar_buffer.mut_buffer(),
               m_scalar_buffer.buffer(),
               scalars::to_kernel_arg(other));
    }

    return result;
}

Vector Vector::sdiv(const scalars::Scalar& other) const
{

    if (other.is_zero()) { throw std::domain_error("division by zero"); }

    scalars::Scalar recip(scalar_type(), 1);
    recip /= other;

    return right_smul(recip);
}

Vector& Vector::add_inplace(const Vector& other)
{
    if (&other != this) {
        apply_binary_kernel("add_inplace", *this, other);
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
        apply_binary_kernel("sub_inplace", *this, other);
    } else {
        // Subtracting a vector from itself yields zero
        this->set_zero();
    }

    return *this;
}

Vector& Vector::smul_inplace(const scalars::Scalar& other)
{
    // bad implementation, but it will do for now
    const Vector tmp = this->right_smul(other);
    *this = tmp;
    return *this;
}

Vector& Vector::sdiv_inplace(const scalars::Scalar& other)
{
    // bad implementation, but it will do for now.
    const Vector tmp = this->sdiv(other);
    *this = tmp;
    return *this;
}

Vector& Vector::add_scal_mul(const Vector& other, const scalars::Scalar& scalar)
{
    if (&other != this) {
        scalars::Scalar tmp(scalar.type_info(), scalar.pointer());
        apply_binary_kernel("add_scal_mul", *this, other, tmp);
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
        scalars::Scalar tmp(scalar.type_info(), scalar.pointer());
        apply_binary_kernel("sub_scal_mul", *this, other, tmp);
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
        auto tmp = scalar.reciprocal();
        apply_binary_kernel("add_scal_mul", *this, other, std::move(tmp));
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
        auto tmp = scalar.reciprocal();
        apply_binary_kernel("sub_scal_mul", *this, other, std::move(tmp));
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

    auto kernel = get_kernel(OperationType::Comparison, "equals", "");
    auto params = get_kernel_launch_params();

    int result = 0;
    devices::KernelArgument result_param(devices::type_info<int>(), &result);
    // TODO: handle sparse vectors.
    kernel(params,
           std::move(result_param),
           m_scalar_buffer.buffer(),
           other.m_scalar_buffer.buffer());

    return false;
}

std::ostream& algebra::operator<<(std::ostream& os, const Vector& value)
{
    const auto basis = value.basis();
    for (const auto& item : value) {
        os << item->second << '(' << basis->to_string(item->first) << ')';
    }
    return os;
}
