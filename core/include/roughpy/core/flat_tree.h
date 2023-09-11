// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 10/09/23.
//

#ifndef ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_FLAT_TREE_H_
#define ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_FLAT_TREE_H_

#include "macros.h"
#include "slice.h"
#include "traits.h"
#include "types.h"

#include <iterator>
#include <vector>

namespace rpy {

namespace dtl {

template <typename I>
class LeafIterator;

template <typename I>
class LeafIterable;

template <typename I>
class BranchProxy;


}// namespace dtl

template <typename I=idimn_t>
class FlatBinaryTree : std::vector<I>
{
    using base_t = std::vector<I>;
    using base_iterator = typename base_t::const_iterator;

    I m_nroots;

    static_assert(
            is_integral<I>::value && is_signed<I>::value,
            "FlatTree can only be constructed with signed integral "
            "types"
    );

    static constexpr bool is_branch(I index) noexcept { return index < 0; }
    static constexpr bool is_leaf(I index) noexcept { return index >= 0; }

    bool is_root(I index) const noexcept { return index < m_nroots; }

    friend class dtl::LeafIterator<I>;
    friend class dtl::BranchProxy<I>;

public:
    using index_type = I;
    using size_type = dimn_t;
    using node_iterator = typename base_t::const_iterator;


    explicit FlatBinaryTree(I n_roots)
        : base_t(3*n_roots), m_nroots(n_roots)
    {
        for (I i=0; i<n_roots; ++i) {
            (*this)[i] = -(n_roots + i);
        }
    }

    using base_t::clear;
    using base_t::reserve;

    RPY_NO_DISCARD
    node_iterator node(size_type index) noexcept {
        return base_t::begin() + index;
    }

    RPY_NO_DISCARD
    pair<I, I> children(size_type index) noexcept
    {
        RPY_DBG_ASSERT(index < base_t::size() && is_branch(index));
        return {(*this)[index], (*this)[index] + 1};
    }

    RPY_NO_DISCARD
    I value(size_type index) noexcept
    {
        RPY_DBG_ASSERT(index < base_t::size() && is_leaf(index));
        return (*this)[index];
    }

    RPY_NO_DISCARD
    dtl::LeafIterable<I> leaves(size_type root) const {
        return dtl::LeafIterable<I>(base_t::begin() + root);
    }

    RPY_NO_DISCARD
    dtl::BranchProxy<I> make_branch(node_iterator leaf_node) {
        RPY_CHECK(is_leaf(*leaf_node));
        *leaf_node = -static_cast<I>(base_t::cend() - leaf_node);
        base_t::push_back(0);
        base_t::push_back(0);
        return dtl::BranchProxy<I>(leaf_node);
    }

};

namespace dtl {

template <typename I>
class LeafIterator
{
    using base_t = std::vector<I>;
    using tree_t = FlatBinaryTree<I>;
    using base_iterator = typename base_t::const_iterator;

    std::vector<base_iterator> m_branch_stack;
    base_iterator m_current;

    void climb_branch()
    {
        while (tree_t::is_branch(*m_current)) {
            m_branch_stack.push_back(m_current);
            m_current += -*m_current;
        }
    }

public:
    typedef std::forward_iterator_tag iterator_category;
    typedef ptrdiff_t difference_type;
    typedef I value_type;
    typedef const I& reference;
    typedef base_iterator pointer;

    LeafIterator(base_iterator current) : m_current(current)
    {
        /*
         * Assume current is a valid iterator
         *
         * First we need to walk down the branch until we get to the first leaf
         * below current. That is where the iterator starts.
         *
         * The iterator is considered valid if the branch_stack is non-empty.
         */
        climb_branch();
        RPY_DBG_ASSERT(tree_t::is_leaf(*m_current));
    }

    LeafIterator& operator++()
    {
        if (!m_branch_stack.empty()) {
            RPY_DBG_ASSERT(tree_t::is_leaf(*m_current));
            auto root = m_branch_stack.back();
            if (m_current == root - (*root)) {
                ++m_current;
            } else {
                const auto& back = m_branch_stack.back();
                while (!m_branch_stack.empty() && root != back - (*back)) {
                    m_current = root;
                    root = back;
                    m_branch_stack.pop_back();
                    back = m_branch_stack.back();
                }

                if (!m_branch_stack.empty()) {
                    RPY_DBG_ASSERT(m_current == back - (*back));
                    ++m_current;
                    climb_branch();
                }
            }
        }
        return *this;
    }

    RPY_NO_DISCARD
    const LeafIterator operator++(int)
    {
        auto prev(*this);
        this->operator++();
        return prev;
    }

    RPY_NO_DISCARD
    reference operator*() const noexcept { return *m_current; }
    RPY_NO_DISCARD
    pointer operator->() const noexcept { return m_current; }

    bool operator==(const LeafIterator& other) const noexcept
    {
        return (m_current == other.m_current
                || (m_branch_stack.empty() && other.m_branch_stack.empty()));
    }

    bool operator!=(const LeafIterator& other) const noexcept
    {
        return !operator==(other);
    }
};

template <typename I>
class LeafIterable
{
    using base_iterator = typename std::vector<I>::const_iterator;

    base_iterator m_root;

public:
    using const_iterator = LeafIterator<I>;

    explicit LeafIterable(base_iterator root) : m_root(root) {}

    RPY_NO_DISCARD
    const_iterator begin() const { return const_iterator(m_root); }
    RPY_NO_DISCARD
    const_iterator cbegin() const { return const_iterator(m_root); }
    RPY_NO_DISCARD
    const_iterator end() const { return const_iterator(); }
    RPY_NO_DISCARD
    const_iterator cend() const { return const_iterator(); }
};


template <typename I>
class BranchProxy {
    using base_t = typename FlatBinaryTree<I>::base_t;
    using node_iterator = typename base_t::const_iterator;

    node_iterator m_branch;

public:

    explicit BranchProxy(node_iterator branch) : m_branch(branch)
    {
        RPY_DBG_ASSERT(base_t::is_branch(*branch));
    }

    RPY_NO_DISCARD
    I& left_child() noexcept { return m_branch - *m_branch; }
    RPY_NO_DISCARD
    I& right_child() noexcept { return m_branch - *m_branch + 1; }


};


}// namespace dtl

}// namespace rpy

#endif// ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_FLAT_TREE_H_
