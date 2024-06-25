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
    BasisPointer p_basis = nullptr;

    friend class MutableVectorElement;

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
     * @param index_hint (optional) A hint for the index of the element to
     * optimize the search. If not provided, a default hint is used.
     *
     * @note This function assumes that the vector is already initialized and
     * contains the basis key.
     */
    void delete_element(const BasisKey& key, optional<dimn_t> index_hint);

    explicit Vector(BasisPointer basis, VectorDataPtr base, VectorDataPtr fibre)
        : ScalarVector(std::move(base), std::move(fibre)),
          p_basis(std::move(basis))
    {}

    RPY_NO_DISCARD static bool basis_compatibility_check(const Basis& basis
    ) noexcept
    {
        return true;
    }

public:
    Vector();

    ~Vector();

    Vector(const Vector& other);

    Vector(Vector&& other) noexcept;

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
        : p_data(new VectorData(scalar.type())),
          p_basis(std::move(basis))
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
        : p_data(new VectorData(scalar_type)),
          p_basis(std::move(basis))
    {}

    Vector(BasisPointer basis,
           scalars::ScalarArray&& scalar_data,
           KeyArray&& key_buffer)
        : p_data(new VectorData(std::move(scalar_data), std::move(key_buffer))),
          p_basis(std::move(basis))
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
        : p_data(new VectorData(
                  scalar_type,
                  basis->dense_dimension(vals.size())
          )),
          p_basis(std::move(basis))
    {
        auto& scalar_vals = p_data->mut_scalars();
        for (auto&& [i, v] : views::enumerate(vals)) { scalar_vals[i] = v; }
    }

    Vector& operator=(const Vector& other);

    Vector& operator=(Vector&& other) noexcept;

    RPY_NO_DISCARD const scalars::ScalarArray& scalars() const noexcept
    {
        return p_data->scalars();
    }
    RPY_NO_DISCARD const KeyArray& keys() const noexcept
    {
        return p_data->keys();
    }
    RPY_NO_DISCARD scalars::ScalarArray& mut_scalars() noexcept
    {
        return p_data->mut_scalars();
    }
    RPY_NO_DISCARD KeyArray& mut_keys() noexcept { return p_data->mut_keys(); }

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
        RPY_CHECK(V::basis_compatibility_check(*p_basis));
        return V(p_basis, p_data, p_fibre);
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
        RPY_CHECK(V::basis_compatibility_check(*p_basis));
        return V(p_basis, p_data, p_fibre);
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
        return p_data->key_buffer().empty();
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
        return !p_data->key_buffer().empty();
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
    RPY_NO_DISCARD BasisPointer basis() const noexcept { return p_basis; }

    /**
     * @brief Retrieves the device of the scalar buffer
     *
     * This method returns the device of the scalar buffer associated with the
     * device object.
     *
     * @return The device of the scalar buffer.
     *
     * @note This method is noexcept, meaning it does not throw any exceptions.
     */
    RPY_NO_DISCARD devices::Device device() const noexcept
    {
        return p_data->scalar_buffer().device();
    }

    /**
     * @brief Retrieves the type of the scalar buffer
     *
     * This method returns the type of the scalar buffer associated with the
     * scalar object.
     *
     * @return The type of the scalar buffer.
     *
     * @note This method is noexcept, meaning it does not throw any exceptions.
     */
    RPY_NO_DISCARD scalars::TypePtr scalar_type() const noexcept
    {
        RPY_DBG_ASSERT(p_data != nullptr);
        return p_data->scalars().type();
    }

    /**
     * @brief Returns the dimension of the vector.
     *
     * The dimension of a vector is the number of elements it contains.
     * This method returns the number of elements in the vector.
     *
     * @return The dimension of the vector.
     *
     * @note This method assumes that the vector is not null.
     * It is the responsibility of the caller to ensure that the vector is not
     * null before calling this method. If the vector is null, the behavior is
     * undefined.
     */
    RPY_NO_DISCARD dimn_t dimension() const noexcept;

    /**
     * @brief Returns the size of the Vector.
     *
     * This method returns the size of the Vector, which represents the number
     * of non-zero elements in the Vector. If the vector is sparse, this number
     * coincides with the dimension.
     *
     * @note This method does not modify the contents of the Vector.
     *
     * @return The size of the Vector as a dimn_t.
     *
     */
    RPY_NO_DISCARD dimn_t size() const noexcept;

    /**
     * @brief Returns true if all elements in the vector are zero; otherwise,
     * returns false.
     *
     * This method checks if all elements in the vector are zero. It first calls
     * the private method `fast_is_zero()` to quickly check if all elements are
     * zero. If `fast_is_zero()` returns false, it then compares the maximum and
     * minimum values of the vector's scalars using the `max` and `min`
     * algorithms from the `scalars::algorithms` namespace. If both the maximum
     * and minimum values are zero, the method returns true; otherwise, it
     * returns false.
     *
     * @return true if all elements in the vector are zero; otherwise, false.
     *
     * @note This method assumes that the scalars of the vector are ordered.
     *
     * @see fast_is_zero()
     * @see scalars::algorithms::max()
     * @see scalars::algorithms::min()
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
    RPY_NO_DISCARD scalars::ScalarCRef get(const BasisKey& key) const;

    /**
     * @brief Get the coefficient of key in the vector mutably
     * @param key Key to query
     * @return Mutable scalar containing coefficient of key
     */
    RPY_NO_DISCARD scalars::ScalarRef get_mut(const BasisKey& key);

    RPY_NO_DISCARD const_iterator begin() const noexcept;
    RPY_NO_DISCARD const_iterator end() const noexcept;

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
    ) const noexcept;

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
        return operator[](p_basis->to_key(index));
    }

    template <typename I>
    RPY_NO_DISCARD enable_if_t<is_integral_v<I>, scalars::Scalar>
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
