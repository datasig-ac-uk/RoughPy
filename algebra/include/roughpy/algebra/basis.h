// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_ALGEBRA_BASIS_H_
#define ROUGHPY_ALGEBRA_BASIS_H_

#include "algebra_fwd.h"

#include <roughpy/core/hash.h>
#include <roughpy/core/smart_ptr.h>
#include <roughpy/core/traits.h>

#include "basis_key.h"
#include "roughpy_algebra_export.h"

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT KeyIteratorState
{
public:
    void* operator new(std::size_t count);
    void operator delete(void* ptr, std::size_t count);

    virtual ~KeyIteratorState();
    virtual void advance() noexcept = 0;

    virtual bool finished() const noexcept = 0;

    virtual BasisKey value() const noexcept = 0;

    virtual bool equals(BasisKey k1, BasisKey k2) const noexcept = 0;
};

namespace dtl {

class KeyRangeIterator
{
    KeyIteratorState* p_state;
    BasisKey m_value;

public:
    using value_type = BasisKey;
    using pointer = const BasisKey*;
    using reference = const BasisKey&;
    using difference_type = idimn_t;
    using iterator_tag = std::forward_iterator_tag;

    KeyRangeIterator() = default;

    explicit KeyRangeIterator(KeyIteratorState* state) : p_state(state)
    {
        update_value();
    }

    KeyRangeIterator& operator++() noexcept
    {
        if (p_state != nullptr) {
            p_state->advance();
            update_value();
        }
        return *this;
    }

    const KeyRangeIterator operator++(int) noexcept
    {
        KeyRangeIterator prev(*this);
        operator++();
        return prev;
    }

private:
    void update_value() noexcept
    {
        if (!m_value && p_state != nullptr && !p_state->finished()) {
            m_value = p_state->value();
        }
    }

public:
    reference operator*() noexcept { return m_value; }

    pointer operator->() noexcept { return &m_value; }

    friend bool operator==(
            const KeyRangeIterator& lhs,
            const KeyRangeIterator& rhs
    ) noexcept
    {
        if (lhs.p_state == nullptr) {
            return rhs.p_state == nullptr || rhs.p_state->finished();
        }
        if (rhs.p_state == nullptr) { return lhs.p_state->finished(); }

        return lhs.p_state == rhs.p_state
                && lhs.p_state->equals(lhs.m_value, rhs.m_value);
    }

    friend bool operator!=(
            const KeyRangeIterator& lhs,
            const KeyRangeIterator& rhs
    ) noexcept
    {
        if (lhs.p_state == nullptr) {
            return rhs.p_state != nullptr && !rhs.p_state->finished();
        }
        if (rhs.p_state == nullptr) { return !lhs.p_state->finished(); }

        return lhs.p_state != rhs.p_state
                || lhs.p_state->equals(lhs.m_value, rhs.m_value);
    }
};

}// namespace dtl

class KeyRange
{
    KeyIteratorState* p_state = nullptr;

public:
    using iterator = dtl::KeyRangeIterator;
    using const_iterator = iterator;

    KeyRange();

    explicit KeyRange(KeyIteratorState* state) noexcept;

    KeyRange(const KeyRange& other) = delete;

    KeyRange(KeyRange&& other) noexcept;

    ~KeyRange();

    KeyRange& operator=(const KeyRange& other) = delete;

    KeyRange& operator=(KeyRange&& other) noexcept;

    const_iterator begin() const noexcept { return const_iterator(p_state); }

    const_iterator end() const noexcept
    {
        (void) this;
        return const_iterator();
    }
};

/**
 * @brief The BasisComparison enum class represents the comparison between two
 * bases
 *
 * The BasisComparison enum class defines three possible comparison results
 * between two bases: IsSame, IsCompatible, and IsNotCompatible. These results
 * are used to determine the relationship between two bases.
 *
 * - IsSame: Indicates that the two bases are identical.
 * - IsCompatible: Indicates that the two bases are compatible, meaning that
 *   they can be used together in certain operations.
 * - IsNotCompatible: Indicates that the two bases are not compatible, meaning
 *   that they cannot be used together in certain operations.
 *
 * The BasisComparison class is typically used in the context of the Basis
 * class, where it is returned by the compare() member function to determine the
 * relationship between two bases.
 */
enum class BasisComparison
{
    IsSame,
    IsCompatible,
    IsNotCompatible
};

/**
 * @brief The Basis class represents a basis of a vector space
 *
 * The Basis class is an abstract base class that represents a basis of a vector
 * space. It provides functions to query information about the basis such as its
 * ID, flags, and various properties related to the keys of the basis.
 *
 * The keys of the basis are objects of type BasisKey, which must derive from
 * BasisKey class. The keys can be used to perform various operations on the
 * basis such as determining equality, computing the hash, and obtaining a
 * string representation.
 *
 * In addition, the Basis class provides functions specific to different types
 * of bases such as ordered bases, graded bases, and word-like bases.
 *
 * To implement a custom basis, you need to derive a class from Basis and
 * override the virtual functions to provide the desired functionality for the
 * specific basis type.
 */
class ROUGHPY_ALGEBRA_EXPORT Basis : public RcBase<Basis>
{
    struct Flags {
        bool is_ordered : 1;
        bool is_graded : 1;
        bool is_word_like : 1;
    };

    const string_view m_basis_id;

    Flags m_flags;

protected:
    Basis(string_view id_string, Flags flags)
        : m_basis_id(id_string),
          m_flags(flags)
    {}

public:
    virtual ~Basis();

    /**
     * @brief Returns the ID of the basis as a string view.
     *
     * This function returns the ID of the basis as a string view. The ID is a
     * unique identifier that represents the basis.
     *
     * @return The ID of the basis as a string view.
     *
     * @note The returned string view is guaranteed to remain valid as long as
     * the basis object is valid. However, the string view should not be used
     * once the basis object is destroyed.
     *
     * @remark This method does not modify the state of the basis object and can
     * be safely called on const instances.
     *
     * @see BasisKey
     */
    RPY_NO_DISCARD string_view id() const noexcept { return m_basis_id; }

    /**
     * @brief Checks if the basis is ordered
     *
     * The is_ordered() method determines whether the basis is ordered.
     * An ordered basis is a basis where the elements have a specific
     * arrangement according to a certain criterion or property.
     *
     * @return  true if the basis is ordered, false otherwise
     *
     * @note This method assumes that the basis has been properly initialized
     *       before calling is_ordered(). Failure to do so may lead to undefined
     *       behavior.
     *
     * @see Basis::initialize()
     */
    RPY_NO_DISCARD bool is_ordered() const noexcept
    {
        return m_flags.is_ordered;
    }

    /**
     * @brief Determines whether the basis is graded
     *
     * This method checks whether the basis is graded. A graded basis is a basis
     * that can be decomposed into a direct sum of graded subspaces, where each
     * subspace corresponds to a different grading value.
     *
     * @return True if the basis is graded, false otherwise.
     *
     * @note This method does not throw any exceptions.
     */
    RPY_NO_DISCARD bool is_graded() const noexcept { return m_flags.is_graded; }

    /**
     * @brief Checks if the basis is word-like
     *
     * The is_word_like() method determines whether the basis is word-like.
     * A word-like basis is a basis where the elements can be represented as
     * words in a language defined by the basis.
     *
     * @return true if the basis is word-like, false otherwise
     *
     * @note This method assumes that the basis has been properly initialized
     * before calling is_word_like(). Failure to do so may lead to undefined
     * behavior.
     *
     * @see Basis::initialize()
     */
    RPY_NO_DISCARD bool is_word_like() const noexcept
    {
        return m_flags.is_word_like;
    }

    /**
     * @brief Checks if the given key exists in the basis
     *
     * This method checks if the given key exists in the basis. The key is
     * passed as a parameter and its type must derive from the BasisKey class.
     *
     * @param key The key to check for existence in the basis
     * @return True if the key exists in the basis, False otherwise
     */
    RPY_NO_DISCARD virtual bool has_key(BasisKey key) const noexcept = 0;

    /**
     * @brief Converts a basis key to a string representation
     *
     * This method converts a given basis key to its string representation. The
     * string representation can be used for various purposes such as display or
     * serialization.
     *
     * @param key The basis key to convert to a string
     * @return The string representation of the basis key
     */
    RPY_NO_DISCARD virtual string to_string(BasisKey key) const = 0;

    /**
     * @brief Determine if two keys are equal
     * @param k1 first key
     * @param k2 second key
     * @return true if both keys belong to the basis and are equal, otherwise
     * false
     */
    RPY_NO_DISCARD virtual bool equals(BasisKey k1, BasisKey k2) const = 0;

    /**
     * @brief Get the hash of a key
     * @param k1 Key to hash
     * @return hash of the key
     */
    RPY_NO_DISCARD virtual hash_t hash(BasisKey k1) const = 0;

    /**
     * @brief Get the max dimension supported by this basis
     * @return maximum dimension of the basis
     *
     * In mathematical terms, this is simply the dimension of the vector space
     * spanned by the basis. However, in this context, a basis might actually
     * represent a family of graded bases and so the dimension might not be a
     * well defined number. The maximum dimension is well defined.
     */
    RPY_NO_DISCARD virtual dimn_t max_dimension() const noexcept;

    /**
     * @brief Returns the dense dimension of the basis
     *
     * This function returns the dense dimension of the basis. The dense
     * dimension represents the number of elements in the basis that are
     * considered dense. The size parameter represents the total number of
     * elements in the basis.
     *
     * @param size The total number of elements in the basis
     *
     * @return The dense dimension of the basis
     */
    RPY_NO_DISCARD virtual dimn_t dense_dimension(dimn_t size) const;

    /*
     * Ordered basis functions
     */

    /**
     * @brief Determines if a basis key k1 is less than another basis key k2
     *
     * This method compares two basis keys, k1 and k2, and determines if k1 is
     * less than k2. The basis keys are objects derived from the BasisKey class.
     *
     * @param k1 The first basis key to compare
     * @param k2 The second basis key to compare
     *
     * @return True if k1 is less than k2, False otherwise.
     *
     * @note This method throws a std::runtime_error if the basis is not
     * ordered.
     */
    RPY_NO_DISCARD virtual bool less(BasisKey k1, BasisKey k2) const;

    /**
     * @brief Converts a BasisKey to its corresponding index in the basis
     *
     * This method converts a BasisKey object to its corresponding index in the
     * basis. The index represents the position of the key within the basis.
     *
     * @param key The BasisKey object to convert to index
     *
     * @return The index of the given BasisKey in the basis
     *
     * @throw std::runtime_error if the basis is not ordered
     */
    RPY_NO_DISCARD virtual dimn_t to_index(BasisKey key) const;

    /**
     * @brief Converts the given index to a BasisKey
     *
     * This method converts the given index to a BasisKey object. The index
     * represents the position of an element in the basis. The returned BasisKey
     * object can be used in various operations related to the basis.
     *
     * Note that this method is virtual and must be overridden in derived
     * classes to provide the desired functionality specific to the basis type.
     *
     * @param index The index of the element in the basis
     *
     * @return The BasisKey object representing the element at the given index
     *
     * @throws std::runtime_error if the basis is not ordered
     */
    RPY_NO_DISCARD virtual BasisKey to_key(dimn_t index) const;

    /**
     * @brief Iterate over the keys of the basis
     *
     * The iterate_keys() method is a virtual method that returns a range of
     * basis keys. It allows iterating over the keys of the basis in a for-each
     * loop or any other loop construct that supports range-based iteration.
     *
     * @return A KeyRange object representing the range of basis keys
     *
     * @throw std::runtime_error when the basis is not ordered
     */
    RPY_NO_DISCARD virtual KeyRange iterate_keys() const;

    /*
     * Graded basis functions
     */

    /**
     * @brief Returns the maximum degree of the basis
     *
     * This function returns the maximum degree of the basis. It is a virtual
     * function that can be overridden by derived classes. If the derived class
     * does not override this function, a runtime error is thrown with the
     * message "basis is not graded".
     *
     * @return The maximum degree of the basis
     * @throws std::runtime_error When the basis is not graded
     */
    RPY_NO_DISCARD virtual deg_t max_degree() const;

    /**
     * @brief Calculate the degree of a basis key
     *
     * This method calculates the degree of the given basis key.
     * The degree represents a measure of the object's complexity or magnitude.
     *
     * @param key The basis key for which to calculate the degree
     *
     * @return The degree of the basis key
     */
    RPY_NO_DISCARD virtual deg_t degree(BasisKey key) const;

    /**
     * @brief Converts a degree to its corresponding dimension in the basis
     *
     * This method converts a given degree to its corresponding dimension in the
     * basis. The dimension represents the number of elements in the basis at
     * the given degree.
     *
     * @param degree The degree for which to determine the dimension
     * @return The dimension of the basis at the given degree
     * @throws std::runtime_error if the basis is not graded
     */
    RPY_NO_DISCARD virtual dimn_t dimension_to_degree(deg_t degree) const;

    /**
     * @brief Iterates over the keys of a specified degree in the basis
     *
     * The iterate_keys_of_degree method iterates over the keys of a specified
     * degree in the basis. This method is applicable only for graded or ordered
     * bases. If the basis is not graded or ordered, an exception of type
     * std::runtime_error is thrown.
     *
     * @param degree The degree of the keys to iterate over
     * @return A KeyRange object representing the range of keys of the specified
     * degree
     * @throws std::runtime_error if the basis is not graded or ordered
     */
    RPY_NO_DISCARD virtual KeyRange iterate_keys_of_degree(deg_t degree) const;

    /*
     * Word like basis functions
     */

    /**
     * @brief Get the size of the alphabet for the basis
     *
     * This method returns the size of the alphabet for the basis. It is a
     * virtual method that is meant to be overridden by derived classes. If the
     * basis is not word-like, this method throws a std::runtime_error.
     *
     * @return The size of the alphabet for the basis
     *
     * @throws std::runtime_error if the basis is not word-like
     */
    RPY_NO_DISCARD virtual deg_t alphabet_size() const;

    /**
     * @brief Check if a BasisKey represents a letter in a word-like basis
     *
     * This method is used to check if a given BasisKey object represents a
     * letter in a word-like basis. The word-like basis is a type of basis where
     * the keys represent letters in a word.
     *
     * @param key The BasisKey object to be checked
     *
     * @return True if the given BasisKey represents a letter in a word-like
     * basis, false otherwise
     *
     * @see BasisKey
     * @see Basis
     * @see is_word_like
     * @see is_ordered
     * @see is_graded
     *
     * @note This method throws a std::runtime_error if the basis is not
     * word-like.
     */
    RPY_NO_DISCARD virtual bool is_letter(BasisKey key) const;

    /**
     * @brief Get the letter corresponding to the given basis key
     *
     * This method returns the letter corresponding to the given basis key. It
     * is a virtual method that needs to be overridden in derived classes. If a
     * derived class does not override this method, it will throw a
     * std::runtime_error with the message "basis is not word-like".
     *
     * @param key The basis key for which to get the letter
     * @return The letter corresponding to the given basis key
     * @throws std::runtime_error if the basis is not word-like and the method
     * is not overridden
     */
    RPY_NO_DISCARD virtual let_t get_letter(BasisKey key) const;

    /**
     * @brief Returns the parents of a basis key
     *
     * This function returns a pair of optional basis keys that represent the
     * parents of a given basis key. The parents are derived from the current
     * basis key and are used to determine relationships between basis elements
     * in a vector space.
     *
     * @param key The basis key for which to find the parents
     * @return A pair of optional basis keys representing the parents
     *
     * @note This method assumes that the basis is word-like. If this is not the
     * case, a runtime_error will be thrown with an appropriate error message.
     */
    RPY_NO_DISCARD virtual pair<optional<BasisKey>, optional<BasisKey>>
    parents(BasisKey key) const;

    /**
     * @brief Compares the basis with another basis.
     *
     * This method compares the basis with another basis and returns the result
     * as a BasisComparison.
     *
     * @param other A pointer to the other basis to compare with.
     * @return The result of the basis comparison as a BasisComparison enum.
     */
    RPY_NO_DISCARD virtual BasisComparison compare(BasisPointer other
    ) const noexcept;
};

/**
 * @brief Compares two basis pointers
 *
 * This method compares two basis pointers, lhs and rhs, and returns the result
 * of the comparison.
 *
 * The method first checks if both lhs and rhs are non-null. If so, it calls the
 * compare method of lhs and passes rhs as the argument. The compare method is
 * expected to perform the actual comparison and return the result. If either
 * lhs or rhs is null, then the method returns BasisComparison::IsNotCompatible
 * indicating that the two bases are not compatible for comparison.
 *
 * @param lhs The left-hand side basis pointer
 * @param rhs The right-hand side basis pointer
 * @return The result of the comparison as a BasisComparison enum value
 */
inline BasisComparison compare(BasisPointer lhs, BasisPointer rhs)
{

    return (lhs && rhs) ? lhs->compare(rhs) : BasisComparison::IsNotCompatible;
}

// Just for completeness, declare these functions again
ROUGHPY_ALGEBRA_EXPORT void intrusive_ptr_add_ref(const Basis* ptr) noexcept;

ROUGHPY_ALGEBRA_EXPORT void intrusive_ptr_release(const Basis* ptr) noexcept;

/**
 * @brief The KeyHash struct represents a hash function for BasisKey objects
 *
 * The KeyHash struct provides a hash function for BasisKey objects. It takes
 * a const pointer to a Basis object as a member variable. The hash function
 * can be used to compute the hash value for a BasisKey object. It delegates the
 * hashing to the hash function implemented in the Basis class, using the given
 * Basis object.
 *
 * To use the KeyHash struct, create an instance and assign a Basis object to
 * its p_basis member variable. Then, use the () operator to compute the hash
 * value for a BasisKey object.
 *
 * Example usage:
 * @code{.cpp}
 * Basis basis;
 * KeyHash keyHash;
 * keyHash.p_basis = &basis;
 * hash_t hashValue = keyHash(someBasisKey);
 * @endcode
 *
 * @see Basis, BasisKey
 */
struct KeyHash {
    const Basis* p_basis;

    hash_t operator()(const BasisKey& arg) const { return p_basis->hash(arg); }
};

/**
 * @brief The KeyEquals struct represents a comparison function object for
 * BasisKey objects
 *
 * The KeyEquals struct is a comparison function object that is used to compare
 * two BasisKey objects for equality. It is designed to be used in conjunction
 * with containers and algorithms that require comparators, such as
 * std::unordered_map or std::sort.
 *
 * The KeyEquals struct holds a pointer to a Basis object, which is used to
 * access the equals() method of the Basis class. This allows KeyEquals to
 * delegate the equality comparison to the appropriate basis implementation.
 *
 * Example Usage:
 * @code
 * Basis basis; // Assuming a Basis object has been created
 * KeyEquals comparator;
 * comparator.p_basis = &basis;
 *
 * BasisKey key1, key2; // Assuming two BasisKey objects have been created
 * bool equal = comparator(key1, key2);
 * @endcode
 */
struct KeyEquals {
    const Basis* p_basis;

    bool operator()(const BasisKey& left, const BasisKey& right) const
    {
        return p_basis->equals(left, right);
    }
};

BasisPointer tensor_product_basis(
        Slice<BasisPointer> bases,
        std::function<dimn_t(BasisKey)> index_function = nullptr,
        std::function<BasisKey(dimn_t)> key_function = nullptr
);

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_H_
