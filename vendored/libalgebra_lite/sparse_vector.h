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
// Created by user on 26/07/22.
//

#ifndef LIBALGEBRA_LITE_SPARSE_VECTOR_H
#define LIBALGEBRA_LITE_SPARSE_VECTOR_H

#include "implementation_types.h"

#include <iterator>
#include <map>
#include <type_traits>
#include <utility>

#include "basis_traits.h"
#include "coefficients.h"
#include "vector_base.h"

namespace lal {
namespace dtl {

#define LAL_MUTABLE_REF_iOP(OP)                                                \
    template <typename Scalar>                                                 \
    Self& operator OP(Scalar arg) noexcept(noexcept(m_tmp OP arg))             \
    {                                                                          \
        m_tmp OP arg;                                                          \
        return *this;                                                          \
    }

#define LAL_MUTABLE_REF_COMPARE(OP)                                            \
    template <typename Scalar>                                                 \
    bool operator OP(Scalar arg) noexcept(noexcept(m_tmp OP arg))              \
    {                                                                          \
        return m_tmp OP arg;                                                   \
    }

template <typename Vector>
class sparse_mutable_reference
{
    using map_type = typename Vector::map_type;
    using iterator_type = typename map_type::iterator;

    Vector& m_vector;
    iterator_type m_it;
    typename Vector::key_type m_key;
    typename Vector::scalar_type m_tmp;

    using Self = sparse_mutable_reference;

public:
    using key_type = typename Vector::key_type;
    using scalar_type = typename Vector::scalar_type;

    sparse_mutable_reference(Vector& vect, iterator_type it)
        : m_vector(vect), m_it(it), m_key(it->first), m_tmp(it->second)
    {
        assert(it != m_vector.m_data.end());
    }

    sparse_mutable_reference(Vector& vect, const key_type& key)
        : m_vector(vect), m_it(vect.m_data.find(key)), m_key(key), m_tmp(0)
    {
        if (m_it != m_vector.m_data.end()) { m_tmp = m_it->second; }
    }

    ~sparse_mutable_reference()
    {
        if (m_tmp != scalar_type(0)) {
            if (m_it != m_vector.m_data.end()) {
                m_it->second = m_tmp;
            } else {
                m_vector.insert_new_value(m_key, m_tmp);
            }
        } else if (m_it != m_vector.m_data.end()) {
            m_vector.m_data.erase(m_it);
        }
    }

    operator const scalar_type&(
    ) const noexcept// NOLINT(google-explicit-constructor)
    {
        return m_tmp;
    }
    //
    //    template <typename S>
    //    std::enable_if_t<std::is_constructible<scalar_type, S>::value,
    //    sparse_mutable_reference&> operator=(S val) {
    //       m_tmp = scalar_type(val);
    //        return *this;
    //    }

    LAL_MUTABLE_REF_iOP(=) LAL_MUTABLE_REF_iOP(+=) LAL_MUTABLE_REF_iOP(
            -=
    ) LAL_MUTABLE_REF_iOP(*=) LAL_MUTABLE_REF_iOP(/=) LAL_MUTABLE_REF_iOP(<<=)
            LAL_MUTABLE_REF_iOP(>>=) LAL_MUTABLE_REF_iOP(|=) LAL_MUTABLE_REF_iOP(
                    &=
            ) LAL_MUTABLE_REF_iOP(^=) LAL_MUTABLE_REF_iOP(%=)

                    LAL_MUTABLE_REF_COMPARE(==) LAL_MUTABLE_REF_COMPARE(
                            !=
                    ) LAL_MUTABLE_REF_COMPARE(<) LAL_MUTABLE_REF_COMPARE(<=)
                            LAL_MUTABLE_REF_COMPARE(>) LAL_MUTABLE_REF_COMPARE(
                                    >=
                            )

                                    friend constexpr bool
                                    operator==(
                                            const scalar_type lhs,
                                            const sparse_mutable_reference& rhs
                                    ) noexcept
    {
        return lhs == rhs.m_tmp;
    }
};

#undef LAL_MUTABLE_REF_COMPARE
#undef LAL_MUTABLE_REF_iOP

template <typename Vector, typename Iterator, typename Parent>
class sparse_iterator_base
{
protected:
    Vector* p_vector = nullptr;
    Iterator m_it;

    using traits = std::iterator_traits<Iterator>;

    using key_type = typename traits::value_type::first_type;
    using scalar_type = typename traits::value_type::second_type;

public:
    using difference_type = std::ptrdiff_t;
    using value_type = Parent;
    using reference = Parent&;
    using const_reference = const Parent&;
    using pointer = Parent*;
    using const_pointer = const Parent*;
    using iterator_category = std::forward_iterator_tag;

    sparse_iterator_base() : p_vector(nullptr), m_it() {}

    sparse_iterator_base(Vector* vector, Iterator it)
        : p_vector(vector), m_it(it)
    {
        assert(vector != nullptr);
    }

    sparse_iterator_base(Vector& vector, Iterator it)
        : p_vector(&vector), m_it(it)
    {}

    Parent& operator++() noexcept
    {
        ++m_it;
        return static_cast<Parent&>(*this);
    }
    const Parent operator++(int) noexcept
    {
        Parent result(p_vector, m_it);
        ++m_it;
        return result;
    }

    const Parent& operator*() const noexcept
    {
        return static_cast<const Parent&>(*this);
    }
    const Parent* operator->() const noexcept
    {
        return static_cast<const Parent*>(this);
    }

    bool operator==(const sparse_iterator_base& other) const noexcept
    {
        return m_it == other.m_it;
    }
    bool operator!=(const sparse_iterator_base& other) const noexcept
    {
        return m_it != other.m_it;
    }
};

template <typename Vector, typename Iterator>
class sparse_iterator;

template <typename Vector>
class sparse_iterator<Vector, typename Vector::map_type::iterator>
    : public sparse_iterator_base<
              Vector, typename Vector::map_type::iterator,
              sparse_iterator<Vector, typename Vector::map_type::iterator>>
{
    using base = sparse_iterator_base<
            Vector, typename Vector::map_type::iterator,
            sparse_iterator<Vector, typename Vector::map_type::iterator>>;
    using base_iterator = typename Vector::map_type::iterator;

public:
    using difference_type = std::ptrdiff_t;
    using value_type = sparse_iterator;
    using reference = sparse_iterator&;
    using pointer = sparse_iterator*;
    using iterator_category = std::forward_iterator_tag;

    using value_reference = sparse_mutable_reference<Vector>;

    using base::base;

    const typename base::key_type& key() const noexcept
    {
        assert(base::p_vector != nullptr);
        return base::m_it->first;
    }

    value_reference value() const noexcept
    {
        assert(base::p_vector != nullptr);
        return value_reference(*base::p_vector, base::m_it);
    }
};

template <typename Vector>
class sparse_iterator<Vector, typename Vector::map_type::const_iterator>
    : public sparse_iterator_base<
              Vector, typename Vector::map_type::const_iterator,
              sparse_iterator<
                      Vector, typename Vector::map_type::const_iterator>>
{
    using base = sparse_iterator_base<
            Vector, typename Vector::map_type::const_iterator,
            sparse_iterator<Vector, typename Vector::map_type::const_iterator>>;
    using base_iterator = typename Vector::map_type::const_iterator;

public:
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = sparse_iterator;
    using reference = sparse_iterator&;
    using pointer = sparse_iterator*;

    using base::base;

    const typename base::key_type& key() const noexcept
    {
        return base::m_it->first;
    }
    const typename base::scalar_type& value() const noexcept
    {
        return base::m_it->second;
    }
};

}// namespace dtl

template <typename Basis, typename Coefficients>
class sparse_vector : public vectors::vector_base<Basis, Coefficients>
{
    using vec_base = vectors::vector_base<Basis, Coefficients>;
    using typename vec_base::basis_traits;
    using typename vec_base::coeff_traits;

    friend class dtl::sparse_mutable_reference<sparse_vector>;

public:
    using typename vec_base::basis_pointer;
    using typename vec_base::basis_type;
    using typename vec_base::coefficient_ring;
    using typename vec_base::key_type;
    using typename vec_base::rational_type;
    using typename vec_base::scalar_type;
    using map_type = std::map<key_type, scalar_type>;

private:
    map_type m_data;
    deg_t m_degree = 0;
    using vec_base::p_basis;

    friend class dtl::sparse_iterator<
            sparse_vector, typename map_type::iterator>;
    friend class dtl::sparse_iterator<
            const sparse_vector, typename map_type::const_iterator>;

protected:
    sparse_vector(basis_pointer basis, map_type&& arg)
        : vec_base(basis), m_data(arg)
    {}

public:
    using reference = dtl::sparse_mutable_reference<sparse_vector>;
    using const_reference = const scalar_type&;

    using iterator
            = dtl::sparse_iterator<sparse_vector, typename map_type::iterator>;
    using const_iterator = dtl::sparse_iterator<
            const sparse_vector, typename map_type::const_iterator>;

    template <typename Scalar>
    explicit sparse_vector(
            basis_pointer basis, std::initializer_list<Scalar> args
    )
        : vec_base(basis)
    {
        assert(args.size() == 1);
        m_data[key_type()] = scalar_type(*args.begin());
    }

    template <typename Key, typename Scalar>
    explicit sparse_vector(basis_pointer basis, Key k, Scalar s)
        : vec_base(basis)
    {
        scalar_type tmp(s);
        if (tmp != coefficient_ring::zero()) {
            m_data.insert(std::make_pair(key_type(k), tmp));
            update_degree_for_key(k);
        }
    }

    explicit sparse_vector(basis_pointer basis) : vec_base(basis) {}

    void swap(sparse_vector& right) {
        std::swap(m_data, right.m_data);
        std::swap(m_degree, right.m_degree);
        vec_base::swap(right);
    }

    constexpr dimn_t size() const noexcept { return m_data.size(); }
    constexpr bool empty() const noexcept { return m_data.empty(); }
    constexpr dimn_t dimension() const noexcept { return size(); }
    deg_t degree() const noexcept
    {
        deg_t result = 0;
        for (const auto& item : m_data) {
            auto d = p_basis->degree(item.first);
            if (d > result) { result = d; }
        }
        return result;
    }
    dimn_t capacity() const noexcept
    {
        return basis_traits::max_dimension(*p_basis);
    }

    void update_degree(deg_t degree) noexcept { m_degree = degree; }

    iterator begin() noexcept { return {*this, m_data.begin()}; }
    iterator end() noexcept { return {*this, m_data.end()}; }
    const_iterator begin() const noexcept { return {*this, m_data.begin()}; }
    const_iterator end() const noexcept { return {*this, m_data.end()}; }

    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }

    const_reference operator[](const key_type& key) const noexcept
    {
        auto val = m_data.find(key);
        if (val != m_data.end()) { return val->second; }
        return coefficient_ring::zero();
    }

    reference operator[](const key_type& key) noexcept
    {
        return reference(*this, key);
    }

private:
    template <typename Tag = typename basis_traits::degree_tag>
    std::enable_if_t<std::is_same<Tag, with_degree_tag>::value>
    update_degree_for_key(const key_type& key)
    {
        auto degree = p_basis->degree(key);
        if (m_degree < degree && degree < basis_traits::max_degree(*p_basis)) {
            m_degree = degree;
        }
    }

    template <typename Tag = typename basis_traits::degree_tag>
    std::enable_if_t<std::is_same<Tag, without_degree_tag>::value>
    update_degree_for_key(const key_type&)
    {
        // Do Nothing
    }

public:
    void insert_new_value(const key_type& key, const scalar_type& value)
    {
        m_data[key] = value;
        update_degree_for_key(key);
    }

    void clear() noexcept { m_data.clear(); }

    template <typename UnaryOp>
    sparse_vector unary_op(UnaryOp op) const
    {
        map_type data;
        //        data.reserve(m_data.size());
        const auto& zero = Coefficients::zero();
        for (const auto& item : m_data) {
            auto tmp = op(item.second);
            if (tmp != zero) { data.emplace(item.first, std::move(tmp)); }
        }
        return {p_basis, std::move(data)};
    }
    template <typename UnaryOp>
    sparse_vector& inplace_unary_op(UnaryOp&& op)
    {
        auto tmp = this->unary_op(
            [op = std::forward<UnaryOp>(op)](const scalar_type& arg) {
            auto tmp = arg;
            op(tmp);
            return tmp;
        });
        std::swap(m_data, tmp.m_data);
        return *this;
    }

    template <typename BinOp>
    sparse_vector binary_op(const sparse_vector& rhs, BinOp&& op) const
    {
        sparse_vector tmp(*this);
        tmp.inplace_binary_op(rhs, [op=std::forward<BinOp>(op)](scalar_type& l,
                                                             const
                                                scalar_type& r) {
            l = op(l, r);
        });

        return tmp;
    }

    template <typename BinOp>
    sparse_vector& inplace_binary_op(const sparse_vector& rhs, BinOp op)
    {
        const auto lend = m_data.end();
        auto rit = rhs.m_data.begin();
        const auto rend = rhs.m_data.end();

        const auto& zero = coefficient_ring::zero();

        for (; rit != rend; ++rit) {
            auto it = m_data.find(rit->first);
            if (it != lend) {
                op(it->second, rit->second);
                if (it->second == zero) {
                    m_data.erase(it);
                } else {
                    update_degree_for_key(it->first);
                }
            } else {
                assert(rit->second != zero);
                scalar_type new_val = zero;
                op(new_val, rit->second);
                insert_new_value(rit->first, new_val);
            }
        }

        return *this;
    }

    bool operator==(const sparse_vector& rhs) const noexcept
    {

        if (m_data.size() != rhs.m_data.size()) { return false; }

        for (auto&& ritem : rhs.m_data) {
            auto found = m_data.find(ritem.first);
            if (found == m_data.end() || found->second != ritem.second) {
                return false;
            }
        }

        return true;
    }
};

}// namespace lal

#endif// LIBALGEBRA_LITE_SPARSE_VECTOR_H
