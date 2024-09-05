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
    friend class MutableVectorElement;

    BasisPointer p_basis;

public:
    using iterator = VectorIterator;
    using const_iterator = VectorIterator;

    using Scalar = scalars::Scalar;
    using ScalarCRef = scalars::ScalarCRef;

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
    virtual void set_zero();

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
    virtual void insert_element(BasisKeyCRef key, scalars::Scalar value);

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
    virtual void delete_element(BasisKeyCRef key);

    explicit Vector(ScalarVector&& base, BasisPointer&& basis)
        : ScalarVector(std::move(base)),
          p_basis(std::move(basis))
    {}

public:
    using ScalarType = devices::TypePtr;

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
    explicit Vector(BasisPointer basis, BasisKeyCRef key, Scalar scalar)
        : p_basis(std::move(basis))
    {
        Vector::insert_element(key, std::move(scalar));
    }

    /**
     * @brief Constructs a Vector object with a given basis and scalar type.
     *
     * @param basis Pointer to the basis of the vector.
     * @param scalar_type Pointer to the scalar type of the vector.
     */
    explicit Vector(BasisPointer basis, scalars::TypePtr scalar_type)
        : ScalarVector(std::move(scalar_type)),
          p_basis(std::move(basis))
    {}

    Vector(BasisPointer basis,
           scalars::ScalarArray&& scalar_data,
           KeyArray&& key_buffer)
        : ScalarVector(scalar_data.type()),
          p_basis(std::move(basis))
    {}

    virtual ~Vector();

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
        : ScalarVector(
                  devices::get_type<T>(),
                  basis->dense_dimension(vals.size())
          ),
          p_basis(std::move(basis))
    {
        RPY_DBG_ASSERT(base_data().size() >= vals.size());
        auto& scalar_vals = mut_base_data();
        for (auto&& [i, v] : views::enumerate(vals)) { scalar_vals[i] = v; }
    }

    Vector& operator=(const Vector& other) = default;

    Vector& operator=(Vector&& other) noexcept = default;

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
    RPY_NO_DISCARD virtual bool is_dense() const noexcept { return true; }

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
    RPY_NO_DISCARD bool is_sparse() const noexcept { return !is_dense(); }

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
    RPY_NO_DISCARD const BasisPointer& basis() const noexcept
    {
        return p_basis;
    }

protected:
    RPY_NO_DISCARD virtual optional<dimn_t> get_index(BasisKeyCRef key) const
    {
        RPY_CHECK(p_basis->has_key(key));
        auto index = p_basis->to_index(key);
        if (index < dimension()) { return index; }
        return nullopt;
    }

public:
    RPY_NO_DISCARD virtual const_iterator base_begin() const;
    RPY_NO_DISCARD virtual const_iterator base_end() const;
    RPY_NO_DISCARD virtual const_iterator fibre_begin() const;
    RPY_NO_DISCARD virtual const_iterator fibre_end() const;

    RPY_NO_DISCARD const_iterator begin() const;
    RPY_NO_DISCARD const_iterator end() const;

    RPY_NO_DISCARD scalars::ScalarCRef base_get(BasisKeyCRef key) const;
    RPY_NO_DISCARD scalars::ScalarCRef fibre_get(BasisKeyCRef key) const;

    /**
     * @brief Get the coefficient of key in the vector
     * @param key Key to query
     * @return Non-mutable scalar containing coefficient of key
     */
    RPY_NO_DISCARD scalars::ScalarCRef get(BasisKeyCRef key) const;
    // {
    //     if (const auto index = get_index(std::move(key))) {
    //         return ScalarVector::get(*index);
    //     }
    //     return scalar_type()->zero();
    // }

    RPY_NO_DISCARD scalars::Scalar base_get_mut(BasisKeyCRef key);
    RPY_NO_DISCARD scalars::Scalar fibre_get_mut(BasisKeyCRef key);

    /**
     * @brief Get the coefficient of key in the vector mutably
     * @param key Key to query
     * @return Mutable scalar containing coefficient of key
     */
    RPY_NO_DISCARD scalars::Scalar get_mut(BasisKeyCRef key);
    // {
    //     if (const auto index = get_index(std::move(key))) {
    //         return ScalarVector::get_mut(*index);
    //     }
    // }

    RPY_NO_DISCARD scalars::ScalarCRef operator[](const BasisKey& key) const
    {
        return get(key);
    }

    RPY_NO_DISCARD scalars::Scalar operator[](const BasisKey& key)
    {
        return get_mut(key);
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
    RPY_NO_DISCARD virtual Vector uminus() const;

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
    RPY_NO_DISCARD virtual Vector add(const Vector& other) const;

    /**
     * @brief Subtract a vector from the current vector
     *
     * This method subtracts the given vector from the current vector and
     * returns a new vector that represents the result.
     *
     * @param other The vector to subtract from the current vector
     * @return The resulting vector after subtraction
     */
    RPY_NO_DISCARD virtual Vector sub(const Vector& other) const;

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
    RPY_NO_DISCARD virtual Vector left_smul(ScalarCRef other) const;

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
    RPY_NO_DISCARD virtual Vector right_smul(ScalarCRef other) const;

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
    RPY_NO_DISCARD virtual Vector sdiv(ScalarCRef other) const
    {
        auto recip = scalars::math::reciprocal(other);
        return right_smul(recip);
    }

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
    virtual Vector& add_inplace(const Vector& other);

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
    virtual Vector& sub_inplace(const Vector& other);

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
    virtual Vector& smul_inplace(ScalarCRef other);

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
    Vector& sdiv_inplace(ScalarCRef other)
    {
        const auto recip = devices::math::reciprocal(other);
        return smul_inplace(recip);
    }

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
    virtual Vector& add_scal_mul(const Vector& other, ScalarCRef scalar);

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
    virtual Vector& sub_scal_mul(const Vector& other, ScalarCRef scalar);

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
    Vector& add_scal_div(const Vector& other, ScalarCRef scalar)
    {
        auto recip = scalars::math::reciprocal(scalar);
        return add_scal_mul(other, recip);
    }

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
    Vector& sub_scal_div(const Vector& other, ScalarCRef scalar)
    {
        auto recip = scalars::math::reciprocal(scalar);
        return sub_scal_mul(other, recip);
    }

    RPY_NO_DISCARD virtual bool is_equal(const Vector& other) const noexcept;

    RPY_NO_DISCARD friend bool
    operator==(const Vector& lhs, const Vector& rhs) noexcept
    {
        return lhs.is_equal(rhs);
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
operator*(const V& lhs, scalars::ScalarCRef rhs)
{
    return V(lhs.right_smul(rhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of_v<Vector, V>, V>
operator*(scalars::ScalarCRef lhs, const V& rhs)
{
    return V(rhs.left_smul(lhs));
}

template <typename V>
RPY_NO_DISCARD enable_if_t<is_base_of_v<Vector, V>, V>
operator/(const V& lhs, scalars::ScalarCRef rhs)
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
operator*=(V& lhs, scalars::ScalarCRef rhs)
{
    lhs.smul_inplace(rhs);
    return lhs;
}

template <typename V>
enable_if_t<is_base_of_v<Vector, V>, V&>
operator/=(V& lhs, scalars::ScalarCRef rhs)
{
    lhs.sdiv_inplace(rhs);
    return lhs;
}

namespace dtl {

class VectorIteratorState;

template <typename T>
class IteratorItemProxy
{
    T m_data;

public:
    template <typename... Args>
    IteratorItemProxy(Args&&... args) : m_data(std::forward<Args>(args)...)
    {}

    operator T() && noexcept { return std::move(m_data); }
    operator const T&() const noexcept { return m_data; }

    const T& operator*() const noexcept { return m_data; }
    const T* operator->() const noexcept { return &m_data; }
    operator T&() noexcept { return m_data; }

    T& operator*() noexcept { return m_data; }

    T* operator->() noexcept { return &m_data; }
};

class ROUGHPY_ALGEBRA_EXPORT VectorIteratorState
    : public platform::SmallObjectBase
{
public:
    using value_type = pair<BasisKey, scalars::ScalarCRef>;
    using proxy_type = IteratorItemProxy<value_type>;

    virtual ~VectorIteratorState() = default;

    RPY_NO_DISCARD virtual std::unique_ptr<VectorIteratorState> copy() const
            = 0;
    virtual void advance() noexcept = 0;

    virtual value_type value() const = 0;
    virtual proxy_type proxy() const = 0;

    virtual bool is_same(const VectorIteratorState& other_state) const noexcept
            = 0;
};

template <typename VectorIt, typename KeyIt>
class ConcreteVectorIteratorState : public VectorIteratorState
{
    VectorIt m_vit;
    KeyIt m_kit;

public:
    ConcreteVectorIteratorState(const VectorIt& vit, const KeyIt& kit)
        : m_vit(vit),
          m_kit(kit)
    {}

    ConcreteVectorIteratorState(VectorIt&& vit, KeyIt&& kit)
        : m_vit(std::move(vit)),
          m_kit(std::move(kit))
    {}

    RPY_NO_DISCARD std::unique_ptr<VectorIteratorState> copy() const override;
    void advance() noexcept override;
    RPY_NO_DISCARD
    value_type value() const override;
    RPY_NO_DISCARD
    proxy_type proxy() const override;

    bool is_same(const VectorIteratorState& other_state
    ) const noexcept override;
};

template <typename VectorIt, typename KeyIt>
std::unique_ptr<VectorIteratorState>
make_iterator_state(VectorIt&& vit, KeyIt&& kit)
{
    using type = ConcreteVectorIteratorState<decay_t<VectorIt>, decay_t<KeyIt>>;
    return std::make_unique<type>(
            std::forward<VectorIt>(vit),
            std::forward<KeyIt>(kit)
    );
}

template <typename VectorIt, typename KeyIt>
std::unique_ptr<VectorIteratorState>
ConcreteVectorIteratorState<VectorIt, KeyIt>::copy() const
{
    return make_iterator_state(m_vit, m_kit);
}

template <typename VectorIt, typename KeyIt>
void ConcreteVectorIteratorState<VectorIt, KeyIt>::advance() noexcept
{
    ++m_vit;
    ++m_kit;
}
template <typename VectorIt, typename KeyIt>
VectorIteratorState::value_type
ConcreteVectorIteratorState<VectorIt, KeyIt>::value() const
{
    return {BasisKey(*m_kit), *m_vit};
}
template <typename VectorIt, typename KeyIt>
VectorIteratorState::proxy_type
ConcreteVectorIteratorState<VectorIt, KeyIt>::proxy() const
{
    return {BasisKey(*m_kit), *m_vit};
}
template <typename VectorIt, typename KeyIt>
bool ConcreteVectorIteratorState<VectorIt, KeyIt>::is_same(
        const VectorIteratorState& other_state
) const noexcept
{
    if (const auto* other
        = dynamic_cast<const ConcreteVectorIteratorState*>(&other_state)) {
        return m_vit == other->m_vit && m_kit == other->m_kit;
    }
    return false;
}

}// namespace dtl

class VectorIterator
{
    std::unique_ptr<dtl::VectorIteratorState> m_state ;

public:
    using value_type = pair<BasisKey, scalars::ScalarCRef>;
    using reference = value_type; // dtl::IteratorItemProxy<value_type>;
    using pointer = dtl::IteratorItemProxy<value_type>;
    using difference_type = ptrdiff_t;
    using iterator_tag = std::forward_iterator_tag;

    template <typename VectorIt, typename KeyIt>
    VectorIterator(VectorIt&& vit, KeyIt&& kit)
        : m_state(dtl::make_iterator_state(
                  std::forward<VectorIt>(vit),
                  std::forward<KeyIt>(kit)
          ))
    {}

    VectorIterator() = default;
    VectorIterator(const VectorIterator& other) : m_state(other.m_state->copy())
    {}
    VectorIterator(VectorIterator&& other) noexcept
        : m_state(std::move(other.m_state))
    {}

    VectorIterator& operator=(const VectorIterator& other)
    {
        if (this != &other) { m_state = other.m_state->copy(); }
        return *this;
    }

    VectorIterator& operator=(VectorIterator&& other) noexcept
    {
        if (this != &other) { m_state = std::move(other.m_state); }
        return *this;
    }

    VectorIterator& operator++() noexcept
    {
        m_state->advance();
        return *this;
    }

    RPY_NO_DISCARD VectorIterator operator++(int) noexcept
    {
        VectorIterator result(*this);
        operator++();
        return result;
    }

    RPY_NO_DISCARD reference operator*() const noexcept
    {
        return m_state->value();
    }

    RPY_NO_DISCARD pointer operator->() const noexcept
    {
        return m_state->proxy();
    }

    RPY_NO_DISCARD friend bool
    operator==(const VectorIterator& lhs, const VectorIterator& rhs) noexcept
    {
        if (lhs.m_state && rhs.m_state) {
            return lhs.m_state->is_same(*rhs.m_state);
        }
        return false;
    }
};

RPY_NO_DISCARD inline bool
operator!=(const VectorIterator& lhs, const VectorIterator& rhs) noexcept
{
    return !(lhs == rhs);
}

inline Vector::const_iterator Vector::base_begin() const
{
    return {ScalarVector::begin(), p_basis->keys_begin()};
}
inline Vector::const_iterator Vector::base_end() const
{
    return {ScalarVector::end(), p_basis->keys_end()};
}
inline Vector::const_iterator Vector::fibre_begin() const
{
    return {};
}

inline Vector::const_iterator Vector::fibre_end() const
{
    return {};
}
inline Vector::const_iterator Vector::begin() const
{
    return base_begin();
}
inline Vector::const_iterator Vector::end() const
{
    return base_end();
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_VECTOR_H
