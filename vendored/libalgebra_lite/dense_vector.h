//
// Created by user on 23/07/22.
//

#ifndef LIBALGEBRA_LITE_DENSE_VECTOR_H
#define LIBALGEBRA_LITE_DENSE_VECTOR_H

#include "implementation_types.h"

#include <memory>
#include <type_traits>

#include "basis_traits.h"
#include "coefficients.h"
#include "vector_base.h"
#include "vector_traits.h"

namespace lal {

namespace dtl {

template <typename Basis, typename Scalar, typename Iterator>
class dense_vector_const_iterator;

template <typename Basis, typename Scalar, typename Iterator>
class dense_vector_iterator;

}// namespace dtl

template <typename Basis>
class key_range;

template <
        typename Basis, typename Coefficients,
        template <typename, typename...> class VectorType, typename... Args>
class dense_vector_base : public vectors::vector_base<Basis, Coefficients>
{
    using vec_base = vectors::vector_base<Basis, Coefficients>;
    using typename vec_base::basis_traits;
    using typename vec_base::coeff_traits;

public:
    using typename vec_base::basis_pointer;
    using typename vec_base::basis_type;
    using typename vec_base::coefficient_ring;
    using typename vec_base::key_type;
    using typename vec_base::rational_type;
    using typename vec_base::scalar_type;

private:
    using storage_type = VectorType<scalar_type, Args...>;

    using vec_base::p_basis;
    storage_type m_storage{};
    deg_t m_degree = 0;

public:
    using size_type = typename storage_type::size_type;
    using difference_type = typename storage_type::difference_type;
    using iterator = dtl::dense_vector_iterator<
            Basis, scalar_type, typename storage_type::iterator>;
    using const_iterator = dtl::dense_vector_const_iterator<
            Basis, scalar_type, typename storage_type::const_iterator>;
    using pointer = typename storage_type::pointer;
    using const_pointer = typename storage_type::const_pointer;
    using reference = typename storage_type::reference;
    using const_reference = typename storage_type::const_reference;

    dense_vector_base(basis_pointer basis, key_type k, scalar_type s)
        : vec_base(basis), m_storage()
    {
        auto index = p_basis->key_to_index(k);
        resize(index);
        m_storage[index] = s;
    }

    explicit dense_vector_base(basis_pointer basis) : vec_base(basis) {}

    dense_vector_base(
            basis_pointer basis, std::initializer_list<scalar_type> args
    )
        : vec_base(basis), m_storage(args)
    {
        resize(args.size());
    }

    template <typename InputIt>
    dense_vector_base(basis_pointer basis, InputIt begin, InputIt end)
        : vec_base(basis), m_storage(begin, end)
    {
        resize(m_storage.size());
    }

    explicit dense_vector_base(basis_pointer basis, size_type n)
        : vec_base(basis), m_storage()
    {
        resize(n);
    }

    template <typename S>
    explicit dense_vector_base(basis_pointer basis, size_type n, const S& val)
        : vec_base(basis), m_storage()
    {
        resize(n, val);
    }

    void swap(dense_vector_base& right) {
        std::swap(m_storage, right.m_storage);
        std::swap(m_degree, right.m_degree);
        vec_base::swap(right);
    }

private:
    size_type adjust_size(size_type n) const noexcept
    {
        auto next = basis_traits::get_next_dimension(*p_basis, n);
        return std::min(basis_traits::max_dimension(*p_basis), next.first);
    }

public:
    void update_degree(deg_t degree) noexcept { m_degree = degree; }

    void reserve_exact(size_type n, deg_t degree = 0)
    {
        m_storage.reserve(n);
        m_degree = degree;
    }

    void resize_exact(size_type n, deg_t degree = 0)
    {
        m_storage.resize(n, coefficient_ring::zero());
        m_degree = degree;
    }

    void resize_exact(size_type n, const scalar_type& scalar, deg_t degree = 0)
    {
        m_storage.resize(n, scalar);
        m_degree = degree;
    }

    void reserve(size_type n)
    {
        auto next = basis_traits::get_next_dimension(*p_basis, n);
        reserve_exact(next.first, next.second);
    }
    void resize(size_type n)
    {
        auto next = basis_traits::get_next_dimension(*p_basis, n);
        resize_exact(next.first, next.second);
    }

    template <typename S>
    void resize(size_type n, const S& val)
    {
        auto next = basis_traits::get_next_dimension(*p_basis, n);
        resize_exact(next.first, val, next.second);
    }

    iterator begin() noexcept { return iterator(&*p_basis, m_storage.begin()); }
    iterator end() noexcept { return iterator(&*p_basis, m_storage.end()); }
    const_iterator begin() const noexcept
    {
        return const_iterator(&*p_basis, m_storage.begin());
    }
    const_iterator end() const noexcept
    {
        return const_iterator(&*p_basis, m_storage.end());
    }
    const_iterator cbegin() const noexcept
    {
        return const_iterator(&*p_basis, m_storage.begin());
    }
    const_iterator cend() const noexcept
    {
        return const_iterator(&*p_basis, m_storage.end());
    }

    size_type size() const noexcept
    {
        const auto& zero = Coefficients::zero();
        return std::count_if(
                m_storage.begin(), m_storage.end(),
                [&zero](const scalar_type& s) { return s != zero; }
        );
    }
    constexpr size_type dimension() const noexcept { return m_storage.size(); }
    constexpr deg_t degree() const noexcept { return m_degree; }
    constexpr bool empty() const noexcept { return m_storage.empty(); }

    template <typename Index>
    reference operator[](Index idx) noexcept
    {
        auto key = p_basis->key_to_index(idx);
        if (key >= m_storage.size()) { resize(key); }
        return m_storage[key];
    }

    template <typename Index>
    const_reference operator[](Index idx) const noexcept
    {
        return m_storage[p_basis->key_to_index(idx)];
    }

    pointer as_mut_ptr() noexcept { return m_storage.data(); }
    const_pointer as_ptr() const noexcept { return m_storage.data(); }

    void clear() noexcept(noexcept(m_storage.clear())) { m_storage.clear(); }

    // these need to be implemented in terms of kernels.

    template <typename UnaryOp>
    dense_vector_base unary_op(UnaryOp op) const
    {
        dense_vector_base result(p_basis);
        result.reserve_exact(m_storage.size(), m_degree);

        const auto begin = m_storage.begin();
        const auto end = m_storage.end();

        for (auto it = begin; it != end; ++it) {
            result.m_storage.emplace_back(op(*it));
        }

        return result;
    }

    template <typename UnaryOp>
    dense_vector_base& inplace_unary_op(UnaryOp op)
    {
        const auto begin = m_storage.begin();
        const auto end = m_storage.end();
        for (auto it = begin; it != end; ++it) { op(*it); }
        return *this;
    }

    template <typename BinaryOp>
    dense_vector_base binary_op(const dense_vector_base& arg, BinaryOp op) const
    {
        dense_vector_base result(p_basis);

        const difference_type lhs_size(m_storage.size());
        const difference_type rhs_size(arg.m_storage.size());

        result.reserve_exact(
                std::max(lhs_size, rhs_size), std::max(m_degree, arg.m_degree)
        );

        const auto mid = std::min(lhs_size, rhs_size);
        const auto& zero = coefficient_ring::zero();

        for (difference_type i = 0; i < mid; ++i) {
            result.m_storage.emplace_back(op(m_storage[i], arg.m_storage[i]));
        }

        for (auto i = mid; i < lhs_size; ++i) {
            result.m_storage.emplace_back(op(m_storage[i], zero));
        }

        for (auto i = mid; i < rhs_size; ++i) {
            result.m_storage.emplace_back(op(zero, arg.m_storage[i]));
        }

        return result;
    }

    template <typename InplaceBinaryOp>
    dense_vector_base&
    inplace_binary_op(const dense_vector_base& rhs, InplaceBinaryOp op)
    {
        const difference_type lhs_size(m_storage.size());
        const difference_type rhs_size(rhs.m_storage.size());

        if (rhs_size > lhs_size) { resize_exact(rhs_size, rhs.m_degree); }

        const auto& zero = coefficient_ring::zero();
        const auto mid = std::min(lhs_size, rhs_size);

        for (difference_type i = 0; i < mid; ++i) {
            op(m_storage[i], rhs.m_storage[i]);
        }

        for (auto i = mid; i < lhs_size; ++i) { op(m_storage[i], zero); }

        for (auto i = mid; i < rhs_size; ++i) {
            op(m_storage[i], rhs.m_storage[i]);
        }

        return *this;
    }

    bool operator==(const dense_vector_base& rhs) const noexcept
    {
        auto mid = std::min(m_storage.size(), rhs.m_storage.size());

        for (dimn_t i = 0; i < mid; ++i) {
            if (m_storage[i] != rhs.m_storage[i]) { return false; }
        }

        const auto& zero = coefficient_ring::zero();

        for (dimn_t i = mid; i < m_storage.size(); ++i) {
            if (m_storage[i] != zero) { return false; }
        }

        for (dimn_t i = mid; i < rhs.m_storage.size(); ++i) {
            if (rhs.m_storage[i] != zero) { return false; }
        }

        return true;
    }
};

template <typename Basis, typename Coefficients>
using dense_vector = dense_vector_base<Basis, Coefficients, std::vector>;


namespace dtl {

template <typename KeyRef, typename ScaRef>
class dense_iterator_item
{
    template <typename B, typename C, typename I>
    friend class dense_vector_iterator;

    template <typename B, typename C, typename I>
    friend class dense_vector_const_iterator;

    KeyRef m_key;
    ScaRef m_sca;

    dense_iterator_item(KeyRef key, ScaRef sca) : m_key(key), m_sca(sca) {}

public:
    dense_iterator_item* operator->() noexcept { return this; }

    KeyRef key() const noexcept { return m_key; }
    ScaRef value() const noexcept { return m_sca; }
};

template <typename Basis, typename Coefficients, typename Iterator>
class dense_vector_iterator
{
    using basis_traits = basis_trait<Basis>;
    using key_type = typename basis_traits::key_type;
    using coeff_traits = coefficient_trait<Coefficients>;
    using scalar_type = typename coeff_traits::scalar_type;
    using basis_pointer = lal::basis_pointer<Basis>;
    //    using iterator_category = std::forward_iterator_tag;

    const Basis* p_basis = nullptr;
    Iterator p_data;
    key_type m_key;

    using it_traits = std::iterator_traits<Iterator>;

public:
    using difference_type = std::ptrdiff_t;
    using value_type = dtl::dense_iterator_item<
            const key_type&, typename it_traits::reference>;
    using reference = value_type;
    using pointer = value_type;
    using iterator_category = std::forward_iterator_tag;

    dense_vector_iterator() = default;

    dense_vector_iterator(const Basis* basis, Iterator data)
        : p_basis(basis), p_data(data),
          m_key(basis_traits::index_to_key(*basis, 0))
    {}

    dense_vector_iterator& operator++()
    {
        ++p_data;
        ++m_key;
        return *this;
    }

    const dense_vector_iterator operator++(int)
    {
        auto current(*this);
        operator++();
        return current;
    }

    reference operator*() const noexcept { return {m_key, *p_data}; }

    pointer operator->() const noexcept { return {m_key, *p_data}; }

    bool operator==(const dense_vector_iterator& other) const noexcept
    {
        return p_data == other.p_data;
    }

    bool operator!=(const dense_vector_iterator& other) const noexcept
    {
        return p_data != other.p_data;
    }
};

template <typename Basis, typename Coefficients, typename Iterator>
class dense_vector_const_iterator
{
    using basis_traits = basis_trait<Basis>;
    using key_type = typename basis_traits::key_type;
    using coeff_traits = coefficient_trait<Coefficients>;
    using scalar_type = typename coeff_traits::scalar_type;

    const Basis* p_basis = nullptr;
    Iterator p_data;
    key_type m_key;

    using it_traits = std::iterator_traits<Iterator>;

public:
    using value_type = dtl::dense_iterator_item<
            const key_type&, typename it_traits::reference>;
    using reference = value_type;
    using pointer = value_type;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    dense_vector_const_iterator() = default;

    dense_vector_const_iterator(const Basis* basis, Iterator data)
        : p_basis(basis), p_data(data),
          m_key(basis_traits::index_to_key(*basis, 0))
    {}

    dense_vector_const_iterator& operator++()
    {
        ++p_data;
        basis_traits::advance_key(*p_basis, m_key);
        return *this;
    }

    const dense_vector_const_iterator operator++(int)
    {
        auto current(*this);
        operator++();
        return current;
    }

    reference operator*() const noexcept { return {m_key, *p_data}; }

    pointer operator->() const noexcept { return {m_key, *p_data}; }

    bool operator==(const dense_vector_const_iterator& other) const noexcept
    {
        return p_data == other.p_data;
    }

    bool operator!=(const dense_vector_const_iterator& other) const noexcept
    {
        return p_data != other.p_data;
    }
};

}// namespace dtl

}// namespace lal

#endif// LIBALGEBRA_LITE_DENSE_VECTOR_H
