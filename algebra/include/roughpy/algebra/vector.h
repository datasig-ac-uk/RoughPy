//
// Created by sam on 1/27/24.
//

#ifndef ROUGHPY_ALGEBRA_VECTOR_H
#define ROUGHPY_ALGEBRA_VECTOR_H

#include <roughpy/core/macros.h>
#include <roughpy/core/smart_ptr.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/device_handle.h>
#include <roughpy/devices/kernel.h>
#include <roughpy/scalars/scalar_array.h>

#include <roughpy/scalars/scalar_vector.h>

#include "algebra_fwd.h"
#include "basis.h"
#include "basis_key.h"
#include "key_array.h"

#include <initializer_list>

namespace rpy {
namespace algebra {

class VectorIterator;

class Vector;

class ROUGHPY_ALGEBRA_EXPORT VectorFactory : public platform::SmallObjectBase
{
public:
    virtual ~VectorFactory();

    virtual Vector construct_empty() const = 0;
    virtual Vector construct_from(const scalars::ScalarVector& base) const = 0;
    virtual Vector construct_with_dim(dimn_t dimension) const = 0;
};

class ROUGHPY_ALGEBRA_EXPORT VectorContext : public platform::SmallObjectBase
{
    BasisPointer p_basis;

protected:
    static const VectorContext& get_context(const Vector& vec) noexcept;

public:
    explicit VectorContext(BasisPointer basis) : p_basis(std::move(basis)) {}

    virtual ~VectorContext();

    virtual Rc<VectorContext> empty_like() const noexcept;

    const Basis& basis() const noexcept { return *p_basis; };

    virtual bool is_sparse() const noexcept;

    virtual void resize_by_dim(Vector& dst, dimn_t base_dim, dimn_t fibre_dim);

    virtual void
    resize_for_operands(Vector& dst, const Vector& lhs, const Vector& rhs);

    virtual optional<dimn_t>
    get_index(const Vector& vector, const BasisKey& key) const noexcept;

    RPY_NO_DISCARD virtual Rc<VectorContext> copy() const;

    RPY_NO_DISCARD virtual VectorIterator
    make_iterator(typename scalars::ScalarVector::iterator it) const;

    RPY_NO_DISCARD virtual VectorIterator
    make_const_iterator(typename scalars::ScalarVector::const_iterator it
    ) const;

    virtual dimn_t size(const Vector& vector) const noexcept;
    virtual dimn_t dimension(const Vector& vector) const noexcept;

    virtual void unary_inplace(
            const scalars::UnaryVectorOperation& operation,
            Vector& arg,
            const scalars::ops::Operator& op
    );

    virtual void
    unary(const scalars::UnaryVectorOperation& operation,
          Vector& dst,
          const Vector& arg,
          const scalars::ops::Operator& op);

    virtual void binary_inplace(
            const scalars::BinaryVectorOperation& operation,
            Vector& left,
            const Vector& right,
            const scalars::ops::Operator& op
    );

    virtual void
    binary(const scalars::BinaryVectorOperation& operation,
           Vector& dst,
           const Vector& left,
           const Vector& right,
           const scalars::ops::Operator& op);

    RPY_NO_DISCARD virtual bool
    is_equal(const Vector& left, const Vector& right) const noexcept;
};

/**
 * @brief The Vector class represents a mathematical vector in a given basis.
 *
 * The Vector class provides methods for manipulating and performing operations
 * on vectors. It supports various types of vectors, such as dense and sparse
 * vectors, and allows for inserting, deleting, and accessing elements of the
 * vector. It also provides operations for scalar multiplication, vector
 * addition, and dot product.
 *
 * @see Basis
 * @see VectorData
 */
class ROUGHPY_ALGEBRA_EXPORT Vector : public scalars::ScalarVector
{
    friend class MutableVectorElement;

    Rc<VectorContext> p_context;
    friend class VectorContext;

public:
    using iterator = VectorIterator;
    using const_iterator = VectorIterator;

    using Scalar = scalars::Scalar;

protected:
    /**
     * @brief Sets all the coefficients of the vector to zero.
     *
     * This method sets all the coefficients of the vector to zero. If the
     * vector is dense, it uses a scalar algorithm to efficiently fill the
     * scalar array with zeroes. If the vector is sparse, it clears the scalar
     * array and the key array.
     *
     * @note This method modifies the vector in place.
     */
    void set_zero();

    /**
     * @brief Insert an element into the vector
     *
     * This method allows the user to insert an element into the vector
     * represented by the `Vector` class. The element is specified by a
     * `BasisKey` object and a `scalars::Scalar` value.
     *
     * @param key The `BasisKey` object that specifies the position where the
     * element should be inserted.
     * @param value The `scalars::Scalar` value of the element to be inserted.
     */
    void insert_element(const BasisKey& key, scalars::Scalar value);

    /**
     * @brief Deletes an element from the vector given a basis key and,
     * optionally, an index hint.
     *
     * This function removes an element from the vector identified by the
     * provided basis key and, if supplied, the index hint. The basis key
     * specifies the element to delete, and the index hint can be used to
     * optimize the search for the element.
     *
     * @param key The basis key specifying the element to delete.
     *
     * @note This function assumes that the vector is already initialized and
     * contains the basis key.
     */
    void delete_element(const BasisKey& key);

    RPY_NO_DISCARD static bool basis_compatibility_check(const Basis& basis
    ) noexcept
    {
        return true;
    }

public:
    Vector() = default;

    Vector(const Vector& other) = default;

    Vector(Vector&& other) noexcept = default;

    /**
     * @brief Constructs a Vector object with a given basis, basis key, and
     * scalar value.
     *
     * This constructor initializes a Vector object with the specified basis,
     * basis key, and scalar value. It creates a new VectorData object with the
     * type of the scalar and assigns it to the p_data member variable. The
     * basis is moved into the p_basis member variable.
     *
     * @param basis Pointer to the basis object associated with the vector.
     * @param key Key for the basis.
     * @param scalar Scalar value for the vector.
     */
    explicit Vector(BasisPointer basis, BasisKey key, Scalar scalar)
        : ScalarVector(std::move(scalar.type())),
          p_context(basis->default_vector_context())
    {
        insert_element(std::move(key), std::move(scalar));
    }

    /**
     * @brief Constructs a Vector object with a given basis and scalar type.
     *
     * @param basis Pointer to the basis of the vector.
     * @param scalar_type Pointer to the scalar type of the vector.
     */
    explicit Vector(BasisPointer basis, scalars::TypePtr scalar_type)
        : ScalarVector(std::move(scalar_type)),
          p_context(basis->default_vector_context())
    {}

    Vector(BasisPointer basis,
           scalars::ScalarArray&& scalar_data,
           KeyArray&& key_buffer)
        : ScalarVector(std::move(scalar_data)),
          p_context(basis->default_vector_context())
    {}

    Vector(Rc<VectorContext> context,
           scalars::TypePtr scalar_type,
           dimn_t dim = 0)
        : ScalarVector(std::move(scalar_type), dim),
          p_context(std::move(context))
    {}

    /**
     * @brief Constructs a Vector object with the provided basis, scalar type,
     * and initial values.
     *
     * This constructor initializes a Vector object with the given basis, scalar
     * type, and initial values. The basis represents the mathematical space in
     * which the vector is defined. The scalar type determines the type of
     * values the vector can store. The initial values are specified as an
     * initializer list of type T.
     *
     * @param basis     A pointer to the basis representing the mathematical
     * space.
     * @param scalar_type  A pointer to the scalar type for the vector.
     * @param vals      An initializer list of values for initializing the
     * vector.
     */
    template <typename T>
    Vector(BasisPointer basis,
           scalars::TypePtr scalar_type,
           std::initializer_list<T> vals)
        : ScalarVector(scalar_type, basis->dense_dimension(vals.size())),
          p_context(basis->default_vector_context())
    {
        auto& scalar_vals = mut_base_data();
        for (auto&& [i, v] : views::enumerate(vals)) { scalar_vals[i] = v; }
    }

    Vector& operator=(const Vector& other) = default;

    Vector& operator=(Vector&& other) noexcept = default;

    /**
     * @brief Borrows a vector if the type of V is a subclass of Vector.
     *
     * This method returns a borrowed vector if the type of V is a subclass of
     * Vector. It checks the compatibility of V's basis with the basis stored in
     * this object. If compatibility is confirmed, it creates a new vector of
     * type V using the same basis, data, and fibre as this object and returns
     * it.
     *
     * @tparam V The type of vector to borrow. Must be a subclass of Vector.
     *
     * @return A borrowed vector of type V with the same basis, data, and fibre
     * as this object.
     *
     * @see Vector
     * @see borrow_mut
     */
    template <typename V>
    enable_if_t<is_base_of_v<Vector, V>, V> borrow() const
    {
        RPY_CHECK(V::basis_compatibility_check(p_context->basis()));
        return V(*this, p_context);
    }

    /**
     * @brief Mutably borrows the vector and checks compatibility with the
     * basis.
     *
     * The `borrow_mut` method mutably borrows the vector and checks
     * compatibility with the basis associated with the vector. If the vector is
     * not compatible with the basis, an exception is thrown. The method then
     * returns a mutable reference to the borrowed vector.
     *
     * @tparam V The type of vector.
     * @return A mutable reference to the borrowed vector.
     *
     * @note This method requires that `V` is a derived of `Vector`.
     *
     * @see basis_compatibility_check
     * @see borrow
     */
    template <typename V>
    enable_if_t<is_base_of_v<Vector, V>, V> borrow_mut()
    {
        RPY_CHECK(V::basis_compatibility_check(p_context->basis()));
        return V(*this, p_context);
    }

    /**
     * @brief Checks if the vector is dense.
     *
     * The `is_dense` method determines whether the vector is dense or not. A
     * dense vector is a vector where all the elements are stored contiguously
     * in memory. This is in contrast to a sparse vector, where only non-zero
     * elements are stored.
     *
     * @return `true` if the vector is dense, `false` otherwise.
     */
    RPY_NO_DISCARD bool is_dense() const noexcept
    {
        return p_context ? !p_context->is_sparse() : false;
    }

    /**
     * @brief Checks whether the vector is sparse.
     *
     * The is_sparse() method determines whether the vector is sparse by
     * checking if the key buffer of the vector's data is empty. A sparse vector
     * is one that stores only non-zero elements and uses an index-based lookup,
     * while a dense vector stores all elements, including zeros, and uses
     * contiguous memory for storage.
     *
     * @return True if the vector is sparse, false otherwise.
     */
    RPY_NO_DISCARD bool is_sparse() const noexcept
    {
        return p_context ? p_context->is_sparse() : true;
    }

    /**
     * @brief Retrieves the type of the vector
     *
     * This method returns the type of the vector, which can be either
     * sparse or dense. It determines the vector type based on the
     * sparsity of the vector and returns the corresponding value from
     * the VectorType enumeration.
     *
     * @return The type of the vector, either Sparse or Dense, as a value
     *         from the VectorType enumeration.
     *
     * @note The returned vector type is based on the sparsity of the vector.
     *       If the vector is sparse, the method returns VectorType::Sparse,
     *       otherwise it returns VectorType::Dense.
     * @note This method is noexcept, meaning it does not throw any exceptions.
     */
    RPY_NO_DISCARD VectorType vector_type() const noexcept
    {
        return (is_sparse()) ? VectorType::Sparse : VectorType::Dense;
    }

    /**
     * @brief Get the basis of the vector.
     *
     * This method returns a pointer to the basis of the vector.
     * The basis represents the coordinate system in which the vector is
     * defined.
     *
     * @return A pointer to the basis of the vector.
     */
    RPY_NO_DISCARD BasisPointer basis() const noexcept
    {
        return &p_context->basis();
    }

    /**
     * @brief Get the coefficient of key in the vector
     * @param key Key to query
     * @return Non-mutable scalar containing coefficient of key
     */
    RPY_NO_DISCARD scalars::ScalarCRef get(const BasisKey& key) const;

    /**
     * @brief Get the coefficient of key in the vector mutably
     * @param key Key to query
     * @return Mutable scalar containing coefficient of key
     */
    RPY_NO_DISCARD scalars::ScalarRef get_mut(const BasisKey& key);

    RPY_NO_DISCARD const_iterator begin() const;
    RPY_NO_DISCARD const_iterator end() const;

    /**
     * @brief Gets the index of the given BasisKey in the Vector
     *
     * This method retrieves the index of the specified BasisKey in the Vector.
     *
     * @param key The BasisKey to find the index for
     *
     * @return The index of the BasisKey, or an empty optional if the BasisKey
     * is not found
     * @note The returned index is zero-based.
     *
     * @sa BasisKey
     */
    RPY_NO_DISCARD optional<dimn_t> get_index(const BasisKey& key
    ) const noexcept
    {
        optional<dimn_t> result{};
        if (RPY_LIKELY(p_context)) {
            result = p_context->get_index(*this, key);
        }
        return result;
    }

    RPY_NO_DISCARD scalars::ScalarCRef operator[](const BasisKey& key) const
    {
        return get(key);
    }

    RPY_NO_DISCARD scalars::ScalarRef operator[](const BasisKey& key)
    {
        return get_mut(key);
    }

    template <typename I>
    RPY_NO_DISCARD enable_if_t<is_integral_v<I>, scalars::Scalar>
    operator[](I index) const
    {
        return ScalarVector::get(static_cast<dimn_t>(index));
    }

    template <typename I>
    RPY_NO_DISCARD enable_if_t<is_integral_v<I>, scalars::Scalar>
    operator[](I index)
    {
        return ScalarVector::get_mut(static_cast<dimn_t>(index));
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

public:
    /**
     * @brief Performs the unary minus operation on the vector
     *
     * This method returns a new vector that is the negation of the current
     * vector. The operation is performed element-wise, meaning that each
     * coefficient of the vector will be negated.
     *
     * @return A new vector that is the negation of the current vector
     */
    RPY_NO_DISCARD Vector uminus() const;

    /**
     * @brief Adds another vector to the current vector.
     *
     * This method takes in another vector and adds its corresponding
     * coefficients to the coefficients of the current vector. The operation is
     * performed element-wise and the result is returned as a new vector.
     *
     * @param other The vector to be added.
     * @return A new vector representing the result of the addition.
     */
    RPY_NO_DISCARD Vector add(const Vector& other) const;

    /**
     * @brief Subtract a vector from the current vector
     *
     * This method subtracts the given vector from the current vector and
     * returns a new vector that represents the result.
     *
     * @param other The vector to subtract from the current vector
     * @return The resulting vector after subtraction
     */
    RPY_NO_DISCARD Vector sub(const Vector& other) const;

    /**
     * @brief Left scalar multiplication of a vector.
     *
     * This method performs a left scalar multiplication of the vector with the
     * given scalar.
     *
     * @param other The scalar value to multiply the vector with.
     * @return The resulting vector after left scalar multiplication.
     *
     * @note This method creates a new vector and does not modify the original
     * vector.
     */
    RPY_NO_DISCARD Vector left_smul(const scalars::Scalar& other) const;

    /**
     * @brief Right scalar multiplication of a vector.
     *
     * This method performs right scalar multiplication on the vector.
     * It multiplies each coefficient of the vector by the given scalar and
     * returns a new vector with the result.
     *
     * @param other The scalar to multiply with.
     * @return A new vector resulting from right scalar multiplication.
     */
    RPY_NO_DISCARD Vector right_smul(const scalars::Scalar& other) const;

    /**
     * @brief Perform element-wise division of the vector by a scalar value.
     *
     * This method performs element-wise division of the vector by a scalar
     * value. The resulting vector contains the result of dividing each element
     * of the original vector by the given scalar value.
     *
     * @param other The scalar value to divide the vector by.
     * @return A new Vector object containing the result of the element-wise
     * division.
     */
    RPY_NO_DISCARD Vector sdiv(const scalars::Scalar& other) const;

    /**
     * @brief Adds another vector to the current vector in place.
     *
     * This method adds the given vector `other` to the current vector,
     * modifying the current vector itself. The addition is performed
     * element-wise.
     *
     * @param other The vector to be added.
     * @return Reference to the modified current vector.
     *
     * @note If the `other` vector is the same as the current vector, then the
     * result will be the current vector scaled by 2.
     *
     * @see Vector::add
     */
    Vector& add_inplace(const Vector& other);

    /**
     * @brief Subtract another vector from this vector in place.
     *
     * This method subtracts the coefficients of another vector from the
     * coefficients of this vector in place. The subtraction is performed
     * element-wise, meaning each coefficient in this vector is subtracted by
     * the corresponding coefficient in the other vector.
     *
     * @param other The vector to subtract from this vector.
     *
     * @return A reference to this vector after the subtraction operation.
     *
     * @note If the other vector is the same as this vector, all coefficients in
     * this vector will be set to zero.
     */
    Vector& sub_inplace(const Vector& other);

    /**
     * @brief Multiplies the vector by the given scalar in place
     *
     * This method multiplies each coefficient of the vector by the given scalar
     * value. The vector itself is modified in place, meaning the original
     * vector is changed.
     *
     * @param other The scalar value to multiply the vector coefficients by
     *
     * @return A reference to the modified vector
     *
     * @note This method assumes that the vector and the scalar value are
     * compatible for multiplication.
     */
    Vector& smul_inplace(const scalars::Scalar& other);

    /**
     * @brief Divides the vector element-wise by the given scalar value.
     *
     * This method divides each element of the vector by the given scalar value
     * and updates the vector in-place.
     *
     * @param other The scalar value to divide the vector by.
     *
     * @return A reference to the modified vector.
     */
    Vector& sdiv_inplace(const scalars::Scalar& other);

    /**
     * @brief Adds the scaled product of another vector and a scalar to the
     * current vector.
     *
     * This method adds the scaled product of a given vector and a scalar to the
     * current vector. The given vector and scalar are multiplied element-wise,
     * and the product is then added to each element of the current vector.
     *
     * @param other The vector to be multiplied with the scalar and added to the
     * current vector.
     * @param scalar The scalar by which the vector is scaled before being added
     * to the current vector.
     *
     * @return A reference to the current vector after the addition operation.
     */
    Vector& add_scal_mul(const Vector& other, const scalars::Scalar& scalar);

    /**
     * @brief Subtract another vector scaled by a scalar from the current
     * vector.
     *
     * This method subtracts the specified vector scaled by the specified scalar
     * from the current vector.
     *
     * @param other The vector to be subtracted from the current vector.
     * @param scalar The scalar to scale the other vector.
     *
     * @return A reference to the current vector after the subtraction and
     * scaling operation.
     */
    Vector& sub_scal_mul(const Vector& other, const scalars::Scalar& scalar);

    /**
     * @brief Adds the scaled product of another vector to the current vector.
     *
     * This method adds the scaled product of a given vector and a scalar to the
     * current vector. The scaling is done by dividing by the provided scalar.
     * The given vector and scalar are multiplied element-wise,
     * and the product is then added to each element of the current vector.
     *
     * @param other The vector to be multiplied with the scalar and added to the
     * current vector.
     * @param scalar The scalar by which the vector is scaled before being added
     * to the current vector.
     *
     * @return A reference to the current vector after the addition operation.
     */
    Vector& add_scal_div(const Vector& other, const scalars::Scalar& scalar);

    /**
     * @brief Subtract the scaled product of another vector to the current
     * vector.
     *
     * This method subtracts the scaled product of a given vector and a scalar
     * to the current vector. The scaling is done by dividing by the provided
     * scalar. The given vector and scalar are multiplied element-wise, and the
     * product is then added to each element of the current vector.
     *
     * @param other The vector to be multiplied with the scalar and subtracted
     * from the current vector.
     * @param scalar The scalar by which the vector is scaled before being
     * subtracted from the current vector.
     *
     * @return A reference to the current vector after the subtraction
     * operation.
     */
    Vector& sub_scal_div(const Vector& other, const scalars::Scalar& scalar);

    RPY_NO_DISCARD friend bool
    operator==(const Vector& lhs, const Vector& rhs) noexcept
    {
        return lhs.p_context->is_equal(lhs, rhs);
    }

    RPY_NO_DISCARD friend bool
    operator!=(const Vector& lhs, const Vector& rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

ROUGHPY_ALGEBRA_EXPORT
std::ostream& operator<<(std::ostream& os, const Vector& value);

/*
 * Arithmetic operators are templated so we don't have to reimplement
 * them for any classes that build on top of these.
 */

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of_v<Vector, V>, V> operator-(const V& arg)
{
    return V(arg.uminus());
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of_v<Vector, V>, V>
operator+(const V& lhs, const Vector& rhs)
{
    return V(lhs.add(rhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of_v<Vector, V>, V>
operator-(const V& lhs, const Vector& rhs)
{
    return V(lhs.sub(rhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of_v<Vector, V>, V>
operator*(const V& lhs, const scalars::Scalar& rhs)
{
    return V(lhs.right_smul(rhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of_v<Vector, V>, V>
operator*(const scalars::Scalar& lhs, const V& rhs)
{
    return V(rhs.left_smul(lhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of_v<Vector, V>, V>
operator/(const V& lhs, const scalars::Scalar& rhs)
{
    return V(lhs.sdiv(rhs));
}

template <typename V>
enable_if_t<is_base_of_v<Vector, V>, V&> operator+=(V& lhs, const Vector& rhs)
{
    lhs.add_inplace(rhs);
    return lhs;
}

template <typename V>
enable_if_t<is_base_of_v<Vector, V>, V&> operator-=(V& lhs, const Vector& rhs)
{
    lhs.sub_inplace(rhs);
    return lhs;
}

template <typename V>
enable_if_t<is_base_of_v<Vector, V>, V&>
operator*=(V& lhs, const scalars::Scalar& rhs)
{
    lhs.smul_inplace(rhs);
    return lhs;
}

template <typename V>
enable_if_t<is_base_of_v<Vector, V>, V&>
operator/=(V& lhs, const scalars::Scalar& rhs)
{
    lhs.sdiv_inplace(rhs);
    return lhs;
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_VECTOR_H
