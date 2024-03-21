//
// Created by sam on 1/27/24.
//

#ifndef ROUGHPY_ALGEBRA_VECTOR_H
#define ROUGHPY_ALGEBRA_VECTOR_H

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <roughpy/scalars/devices/buffer.h>
#include <roughpy/scalars/devices/device_handle.h>
#include <roughpy/scalars/devices/kernel.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>

#include "algebra_fwd.h"
#include "basis.h"
#include "key_array.h"

namespace rpy {
namespace algebra {

class Basis;
class BasisKey;

class VectorIterator;

class Vector
{
    scalars::ScalarArray m_scalar_buffer;
    KeyArray m_key_buffer;

    BasisPointer p_basis;

public:
    using iterator = VectorIterator;
    using const_iterator = VectorIterator;

protected:
    /**
     * @brief Reallocate and move contents to a new buffer with the given size
     * @param dim target dimension
     */
    void resize_dim(dimn_t dim);

    /**
     * @brief Reallocate and move contents to a new buffer for the dimension
     * corresponding to the given degree
     * @param degree
     */
    void resize_degree(deg_t degree);

    enum OperationType
    {
        Unary,
        UnaryInplace,
        Binary,
        BinaryInplace,
        Ternary,
        TernaryInplace,
        Comparison
    };

    /**
     * @brief Get the correct kernel for the given operation
     * @param type the type of operation that is required
     * @param operation Operation to get
     * @return Kernel for the required operation
     */
    devices::Kernel
    get_kernel(OperationType type, string_view operation, string_view suffix)
            const;

    devices::KernelLaunchParams get_kernel_launch_params() const;

    /**
     * @brief Get the current size of the buffer
     * @return
     */
    RPY_NO_DISCARD dimn_t buffer_size() const noexcept
    {
        return m_scalar_buffer.size();
    }

    RPY_NO_DISCARD devices::Buffer& mut_scalar_Buffer() noexcept
    {
        return m_scalar_buffer.mut_buffer();
    }
    RPY_NO_DISCARD devices::Buffer& mut_key_buffer() noexcept
    {
        return m_key_buffer;
    }
    RPY_NO_DISCARD const devices::Buffer& scalar_buffer() const noexcept
    {
        return m_scalar_buffer.buffer();
    }
    RPY_NO_DISCARD const devices::Buffer& key_buffer() const noexcept
    {
        return m_key_buffer.buffer();
    }

    RPY_NO_DISCARD bool fast_is_zero() const noexcept
    {
        return p_basis == nullptr || m_scalar_buffer.empty();
    }

    void set_zero();

    optional<dimn_t> get_index(BasisKey key) const noexcept;

public:
    Vector();

    ~Vector();

    Vector(const Vector& other);

    Vector(Vector&& other) noexcept;

    explicit Vector(BasisPointer basis, const scalars::ScalarType* scalar_type)
        : p_basis(std::move(basis)),
          m_scalar_buffer(scalar_type),
          m_key_buffer()
    {}

    Vector(BasisPointer basis,
           scalars::ScalarArray&& scalar_data,
           devices::Buffer&& key_buffer)
        : p_basis(std::move(basis)),
          m_scalar_buffer(std::move(scalar_data)),
          m_key_buffer(std::move(key_buffer))
    {}

    Vector& operator=(const Vector& other);

    Vector& operator=(Vector&& other) noexcept;

    /**
     * @brief Is the vector densely stored
     * @return
     */
    RPY_NO_DISCARD bool is_dense() const noexcept
    {
        return m_key_buffer.is_null();
    }

    /**
     * @brief Is the vector sparse
     * @return
     */
    RPY_NO_DISCARD bool is_sparse() const noexcept
    {
        return !m_key_buffer.is_null();
    }

    RPY_NO_DISCARD VectorType vector_type() const noexcept
    {
        return (is_sparse()) ? VectorType::Sparse : VectorType::Dense;
    }

    /**
     * @brief Get the basis for this vector
     */
    RPY_NO_DISCARD BasisPointer basis() const noexcept { return p_basis; }

    RPY_NO_DISCARD devices::Device device() const noexcept
    {
        return m_scalar_buffer.device();
    }

    RPY_NO_DISCARD const scalars::ScalarType* scalar_type() const noexcept
    {
        auto type = m_scalar_buffer.type();
        RPY_DBG_ASSERT(type);
        return *type;
    }

    /**
     * @brief The total number of elements (including zeros).
     * @return The total number of elements
     */
    RPY_NO_DISCARD dimn_t dimension() const noexcept;

    /**
     * @brief Number of non-zero elements.
     * @return The number of non-zero elements
     */
    RPY_NO_DISCARD dimn_t size() const noexcept;

    /**
     * @brief Quickly check if the vector is zero.
     * @return True if the vector represents zero.
     */
    RPY_NO_DISCARD bool is_zero() const noexcept;

    /**
     * @brief Change the internal representation to dense if possible.
     */
    void make_dense();

    /**
     * @brief Change the internal representation to sparse.
     *
     * Can only fail if the there is a problem with allocation/copying.
     */
    void make_sparse();

    /**
     * @brief Get the coefficient of key in the vector
     * @param key Key to query
     * @return Non-mutable scalar containing coefficient of key
     */
    RPY_NO_DISCARD scalars::Scalar get(BasisKey key) const;

    /**
     * @brief Get the coefficient of key in the vector mutably
     * @param key Key to query
     * @return Mutable scalar containing coefficient of key
     */
    RPY_NO_DISCARD scalars::Scalar get_mut(BasisKey key);

    RPY_NO_DISCARD const_iterator begin() const noexcept;
    RPY_NO_DISCARD const_iterator end() const noexcept;

    RPY_NO_DISCARD scalars::Scalar operator[](BasisKey key) const;

    RPY_NO_DISCARD scalars::Scalar operator[](BasisKey key);

    template <typename I>
    RPY_NO_DISCARD enable_if_t<is_integral<I>::value, scalars::Scalar>
    operator[](I index) const
    {
        return operator[](p_basis->to_key(index));
    }

    template <typename I>
    RPY_NO_DISCARD enable_if_t<is_integral<I>::value, scalars::Scalar>
    operator[](I index)
    {
        return operator[](p_basis->to_key(index));
    }

private:
    /**
     * @brief Check vector compatibility and resize *this
     * @param lhs left reference vector
     * @param rhs  right reference vector
     *
     * This is the binary version of the check for compatibility.
     */
    void check_and_resize_for_operands(const Vector& lhs, const Vector& rhs);

    void apply_binary_kernel(
            string_view kernel_name,
            const Vector& lhs,
            const Vector& rhs,
            optional<scalars::Scalar> multiplier = {}
    );

public:
    RPY_NO_DISCARD Vector uminus() const;

    RPY_NO_DISCARD Vector add(const Vector& other) const;
    RPY_NO_DISCARD Vector sub(const Vector& other) const;
    RPY_NO_DISCARD Vector left_smul(const scalars::Scalar& other) const;
    RPY_NO_DISCARD Vector right_smul(const scalars::Scalar& other) const;
    RPY_NO_DISCARD Vector sdiv(const scalars::Scalar& other) const;

    Vector& add_inplace(const Vector& other);
    Vector& sub_inplace(const Vector& other);

    Vector& smul_inplace(const scalars::Scalar& other);

    Vector& sdiv_inplace(const scalars::Scalar& other);

    Vector& add_scal_mul(const Vector& other, const scalars::Scalar& scalar);
    Vector& sub_scal_mul(const Vector& other, const scalars::Scalar& scalar);
    Vector& add_scal_div(const Vector& other, const scalars::Scalar& scalar);
    Vector& sub_scal_div(const Vector& other, const scalars::Scalar& scalar);

    RPY_NO_DISCARD bool operator==(const Vector& other) const;
    RPY_NO_DISCARD bool operator!=(const Vector& other) const
    {
        return !operator==(other);
    }
};

ROUGHPY_ALGEBRA_EXPORT
std::ostream& operator<<(std::ostream& os, const Vector& value);

/*
 * Arithmetic operators are templated so we don't have to reimplement them for
 * any classes that build on top of these.
 */

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of<Vector, V>::value, V>
operator-(const V& arg)
{
    return V(arg.uminus());
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of<Vector, V>::value, V>
operator+(const V& lhs, const Vector& rhs)
{
    return V(lhs.add(rhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of<Vector, V>::value, V>
operator-(const V& lhs, const Vector& rhs)
{
    return V(lhs.sub(rhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of<Vector, V>::value, V>
operator*(const V& lhs, const scalars::Scalar& rhs)
{
    return V(lhs.right_smul(rhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of<Vector, V>::value, V>
operator*(const scalars::Scalar& lhs, const V& rhs)
{
    return V(rhs.left_smul(lhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of<Vector, V>::value, V>
operator/(const V& lhs, const scalars::Scalar& rhs)
{
    return V(lhs.sdiv(rhs));
}

template <typename V>
enable_if_t<is_base_of<Vector, V>::value, V&>
operator*=(V& lhs, const Vector& rhs)
{
    lhs.add_inplace(rhs);
    return lhs;
}

template <typename V>
enable_if_t<is_base_of<Vector, V>::value, V&>
operator-=(V& lhs, const Vector& rhs)
{
    lhs.sub_inplace(rhs);
    return lhs;
}

template <typename V>
enable_if_t<is_base_of<Vector, V>::value, V&>
operator*=(V& lhs, const scalars::Scalar& rhs)
{
    lhs.smul_inplace(rhs);
    return lhs;
}

template <typename V>
enable_if_t<is_base_of<Vector, V>::value, V&>
operator/=(V& lhs, const scalars::Scalar& rhs)
{
    lhs.sdiv_inplace(rhs);
    return lhs;
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_VECTOR_H
