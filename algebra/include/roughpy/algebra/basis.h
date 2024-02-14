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

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <roughpy/core/hash.h>

#include <roughpy/core/traits.h>

#include "basis_key.h"
#include "roughpy_algebra_export.h"

namespace rpy {
namespace algebra {


class ROUGHPY_ALGEBRA_EXPORT KeyIteratorState {
public:

    virtual void advance() noexcept = 0;

    virtual bool finished() const noexcept = 0;

    virtual BasisKey value() const noexcept = 0;

    virtual bool equals(BasisKey k1, BasisKey k2) const noexcept = 0;
};

namespace dtl {

class KeyRangeIterator {
    KeyIteratorState* p_state;
    BasisKey m_value;
public:

    using value_type = BasisKey;
    using pointer = const BasisKey*;
    using reference = const BasisKey&;
    using difference_type = idimn_t;
    using iterator_tag = std::forward_iterator_tag;

    KeyRangeIterator() = default;

    explicit KeyRangeIterator(KeyIteratorState* state)
        : p_state(state)
    {
        update_value();
    }

    KeyRangeIterator& operator++() noexcept
    {
        if (p_state != nullptr) {
            p_state->advance();
            update_value();
        }
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
    reference operator*() noexcept
    {
        return m_value;
    }

    pointer operator->() noexcept
    {
        return &m_value;
    }


    friend bool operator==(const KeyRangeIterator& lhs,
                           const KeyRangeIterator& rhs) noexcept
    {
        if (lhs.p_state == nullptr) {
            return rhs.p_state == nullptr || rhs.p_state->finished();
        }
        if (rhs.p_state == nullptr) {
            return lhs.p_state->finished();
        }

        return lhs.p_state == rhs.p_state &&
               lhs.p_state->equals(lhs.m_value, rhs.m_value);
    }

    friend bool operator!=(const KeyRangeIterator& lhs,
                           const KeyRangeIterator& rhs) noexcept
    {
        if (lhs.p_state == nullptr) {
            return rhs.p_state != nullptr && !rhs.p_state->finished();
        }
        if (rhs.p_state == nullptr) {
            return !lhs.p_state->finished();
        }

        return lhs.p_state != rhs.p_state ||
               lhs.p_state->equals(lhs.m_value, rhs.m_value);
    }

};


}


class KeyRange {
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


    const_iterator begin() const noexcept
    {
        return const_iterator(p_state);
    }

    const_iterator end() const noexcept
    {
        (void) this;
        return const_iterator();
    }

};


class ROUGHPY_ALGEBRA_EXPORT Basis
    : public boost::intrusive_ref_counter<Basis> {
    struct Flags {
        bool is_ordered: 1;
        bool is_graded: 1;
        bool is_word_like: 1;
    };

    Flags m_flags;

public:

    virtual ~Basis();


    RPY_NO_DISCARD bool is_ordered() const noexcept { return m_flags.is_ordered; }

    RPY_NO_DISCARD bool is_graded() const noexcept { return m_flags.is_graded; }

    RPY_NO_DISCARD bool is_word_like() const noexcept { return m_flags.is_word_like; }

    RPY_NO_DISCARD
    virtual bool has_key(BasisKey key) const noexcept = 0;

    RPY_NO_DISCARD virtual string to_string(BasisKey key) const noexcept = 0;

    /**
     * @brief Determine if two keys are equal
     * @param k1 first key
     * @param k2 second key
     * @return true if both keys belong to the basis and are equal, otherwise false
     */
    RPY_NO_DISCARD
    virtual bool equals(BasisKey k1, BasisKey k2) const noexcept = 0;

    /**
     * @brief Get the hash of a key
     * @param k1 Key to hash
     * @return hash of the key
     */
    RPY_NO_DISCARD
    virtual hash_t hash(BasisKey k1) const noexcept = 0;

    RPY_NO_DISCARD
    virtual dimn_t max_dimension() const noexcept = 0;

    /*
     * Ordered basis functions
     */
    /**
     * @brief Determine if one key precedes another
     * @param k1 first key
     * @param k2 second key
     * @return true if the basis is ordered, has both keys, and k1 precedes k2.
     * Otherwise false.
     */
    RPY_NO_DISCARD virtual bool less(BasisKey k1, BasisKey k2) const noexcept;

    /**
     * @brief Get the index of a basis key in the basis order
     * @param key key to query
     * @return index of key in the basis order if it is ordered, otherwise 0
     */
    RPY_NO_DISCARD
    virtual dimn_t to_index(BasisKey key) const noexcept;

    /**
     * @brief Get the key that corresponds to index
     * @param index index to query
     * @return BasisKey corresponding to index if basis is ordered, and index is
     * valid for this basis, otherwise an invalid key.
     */
    RPY_NO_DISCARD virtual BasisKey to_key(dimn_t index) const noexcept;


    RPY_NO_DISCARD virtual KeyRange iterate_keys() const noexcept;

    /*
     * Graded basis functions
     */

    RPY_NO_DISCARD virtual deg_t max_degree() const noexcept;

    /**
     * @brief Get the degree of a key
     * @param key key to query
     * @return degree of key if basis is graded and key belongs to basis, else 0
     */
    RPY_NO_DISCARD virtual deg_t degree(BasisKey key) const noexcept;

    RPY_NO_DISCARD virtual KeyRange iterate_keys_of_degree(deg_t degree) const noexcept;

    /*
     * Word-like basis functions
     */
    RPY_NO_DISCARD virtual deg_t alphabet_size() const noexcept;

    RPY_NO_DISCARD virtual bool is_letter(BasisKey key) const noexcept;

    RPY_NO_DISCARD virtual let_t get_letter(BasisKey key) const noexcept;

    RPY_NO_DISCARD virtual pair<optional<BasisKey>, optional<BasisKey>> parents(
        BasisKey key) const noexcept;


};


// Just for completeness, declare these functions again
ROUGHPY_ALGEBRA_EXPORT void intrusive_ptr_add_ref(const Basis* ptr) noexcept;

ROUGHPY_ALGEBRA_EXPORT void intrusive_ptr_release(const Basis* ptr) noexcept;


}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_H_
