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

#ifndef LIBALGEBRA_LITE_ALGEBRA_H
#define LIBALGEBRA_LITE_ALGEBRA_H

#include "implementation_types.h"
#include "libalgebra_lite_export.h"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "basis_traits.h"
#include "coefficients.h"
#include "vector.h"
#include "vector_traits.h"

#include "detail/traits.h"

namespace lal {

template <typename Multiplication>
struct multiplication_traits {

    using mult_ptr = std::shared_ptr<const Multiplication>;

private:
    template <typename, typename, typename, typename, typename = void>
    struct has_fma_inplace : false_type {
    };

    template <typename L, typename R, typename F, typename M>
    struct has_fma_inplace<
            L, R, F, M,
            void_t<decltype(M::fma_inplace(
                    declval<L&>(), declval<const R&>(), declval<F>()
            ))>> : true_type {
    };

    template <typename, typename, typename, typename, typename = void>
    struct has_fma_inplace_with_deg : false_type {
    };

    template <typename L, typename R, typename F, typename M>
    struct has_fma_inplace_with_deg<
            L, R, F, M,
            void_t<decltype(M::fma_inplace(declval<L&>(), declval<const R&>(), declval<F>()), declval<deg_t>())>>
        : true_type {
    };

    template <typename V>
    using has_degree_t = is_same<
            typename basis_trait<typename V::basis_type>::degree_tag,
            with_degree_tag>;

public:
    template <typename Result, typename Vector1, typename Vector2, typename Fn>
    static enable_if_t<!has_degree_t<Result>::value> multiply_and_add(
            const Multiplication& mult, Result& result, const Vector1& lhs,
            const Vector2& rhs, Fn fn
    )
    {
        if (lhs.empty() || rhs.empty()) { return; }

        mult.fma(
                result.base_vector(), lhs.base_vector(), rhs.base_vector(), fn
        );
    }

    template <typename Result, typename Vector1, typename Vector2>
    static enable_if_t<!has_degree_t<Result>::value> multiply_and_add(
            const Multiplication& mult, Result& result, const Vector1& lhs,
            const Vector2& rhs
    )
    {
        using scalar_type = typename Result::scalar_type;
        mult.fma(
                result.base_vector(), lhs.base_vector(), rhs.base_vector(),
                [](scalar_type s) { return s; }
        );
    }

    template <typename Result, typename Vector1, typename Vector2, typename Fn>
    static enable_if_t<has_degree_t<Result>::value> multiply_and_add(
            const Multiplication& mult, Result& result, const Vector1& lhs,
            const Vector2& rhs, Fn fn
    )
    {
        using traits = basis_trait<typename Result::basis_type>;
        mult.fma(
                result.base_vector(), lhs.base_vector(), rhs.base_vector(), fn,
                traits::max_degree(result.basis())
        );
    }

    template <typename Result, typename Vector1, typename Vector2>
    static enable_if_t<has_degree_t<Result>::value> multiply_and_add(
            const Multiplication& mult, Result& result, const Vector1& lhs,
            const Vector2& rhs
    )
    {
        using scalar_type = typename Result::scalar_type;
        using traits = basis_trait<typename Result::basis_type>;
        mult.fma(
                result.base_vector(), lhs.base_vector(), rhs.base_vector(),
                [](scalar_type s) { return s; },
                traits::max_degree(result.basis())
        );
    }
    //
    //    template <typename Left, typename Right, typename Fn, typename
    //    Mult=Multiplication> static enable_if_t<!has_degree_t<Left>::value &&
    //    has_fma_inplace<Left, Right, Fn, Mult>::value>
    //    multiply_and_add_inplace(const Multiplication &mult, Left &lhs, const
    //    Right &rhs, Fn fn) {
    //        mult.fma_inplace(lhs.vector_type(), rhs.vector_type(), fn);
    //    }
    //
    //    template <typename Left, typename Right, typename Fn, typename
    //    Mult=Multiplication> static enable_if_t<has_degree_t<Left>::value &&
    //    has_fma_inplace<Left, Right, Fn, Mult>::value>
    //    multiply_and_add_inplace(const Multiplication& mult, Left& lhs, const
    //    Right& rhs, Fn fn)
    //    {
    //        using traits = basis_trait<typename Left::basis_type>;
    //        mult.fma_inplace(lhs.base_vector(), rhs.base_vector(), fn,
    //                traits::max_degree(lhs.basis()));
    //    }
    //
    //    template <typename Left, typename Right, typename Fn, typename
    //    Mult=Multiplication> static enable_if_t<has_degree_t<Left>::value &&
    //    has_fma_inplace<Left, Right, Fn, Mult>::value>
    //    multiply_and_add_inplace(const Multiplication &mult, Left &lhs, const
    //    Right &rhs, Fn fn, deg_t max_deg) {
    //        mult.fma_inplace(lhs.base_vector(), rhs.base_vector(), fn,
    //        max_deg);
    //    }

    template <typename Left, typename Right, typename Fn>
    static enable_if_t<!has_degree_t<Left>::value> multiply_inplace(
            const Multiplication& mult, Left& lhs, const Right& rhs, Fn fn
    )
    {
        //        Left tmp(lhs.get_basis());
        //        mult.fma(tmp.base_vector(), lhs.base_vector(),
        //        rhs.base_vector(), fn); lhs.swap(tmp);
        mult.multiply_inplace(lhs.base_vector(), rhs.base_vector(), fn);
    }

    template <typename Left, typename Right, typename Fn>
    static enable_if_t<has_degree_t<Left>::value> multiply_inplace(
            const Multiplication& mult, Left& lhs, const Right& rhs, Fn fn,
            deg_t max_deg
    )
    {
        //        Left tmp(lhs.get_basis());
        //        mult.fma(tmp.base_vector(), lhs.base_vector(),
        //        rhs.base_vector(), fn, max_deg); std::swap(lhs, tmp);
        //        lhs.swap(tmp);
        mult.multiply_inplace(
                lhs.base_vector(), rhs.base_vector(), fn, max_deg
        );
    }

    template <typename Left, typename Right, typename Fn>
    static enable_if_t<!has_degree_t<Left>::value> multiply_inplace(
            const Multiplication& mult, Left& lhs, const Right& rhs, Fn fn,
            deg_t
    )
    {
        //        Left tmp(lhs.get_basis());
        //        mult.fma(tmp.base_vector(), lhs.base_vector(),
        //        rhs.base_vector(), fn, max_deg); std::swap(lhs, tmp);
        //        lhs.swap(tmp);
        mult.multiply_inplace(lhs.base_vector(), rhs.base_vector(), fn);
    }

    template <typename Left, typename Right, typename Fn>
    static enable_if_t<has_degree_t<Left>::value> multiply_inplace(
            const Multiplication& mult, Left& lhs, const Right& rhs, Fn fn
    )
    {
        auto max_deg
                = basis_trait<typename Left::basis_type>::max_degree(lhs.basis()
                );
        multiply_inplace(mult, lhs, rhs, fn, max_deg);
    }
};

namespace dtl {

template <typename Basis, typename Coefficients>
class general_multiplication_helper
{
protected:
    using basis_traits = basis_trait<Basis>;
    using key_type = typename basis_traits::key_type;
    using coeff_traits = coefficient_trait<Coefficients>;
    using scalar_type = typename coeff_traits::scalar_type;

    using key_value = pair<key_type, scalar_type>;
    std::vector<key_value> right_buffer;

public:
    using const_iterator = typename std::vector<key_value>::const_iterator;

    template <typename Vector>
    explicit general_multiplication_helper(const Vector& rhs) : right_buffer()
    {
        right_buffer.reserve(rhs.size());
        for (auto item : rhs) {
            right_buffer.emplace_back(item.key(), item.value());
        }
    }

    const_iterator begin() const noexcept { return right_buffer.begin(); }
    const_iterator end() const noexcept { return right_buffer.end(); }
};

template <typename It>
class degree_range_iterator_helper
{
    It m_begin, m_end;

public:
    using iterator = It;

    degree_range_iterator_helper(It begin, It end) : m_begin(begin), m_end(end)
    {}

    iterator begin() noexcept { return m_begin; }
    iterator end() noexcept { return m_end; }
};

template <typename Basis, typename Coefficients>
class graded_multiplication_helper
    : protected general_multiplication_helper<Basis, Coefficients>
{
    using base_type = general_multiplication_helper<Basis, Coefficients>;
    using ordering = typename base_type::basis_traits::kv_ordering;

    using base_type::right_buffer;
    using typename base_type::basis_traits;
    using typename base_type::key_type;
    using typename base_type::key_value;
    using typename base_type::scalar_type;

    using base_iter = typename std::vector<key_value>::const_iterator;

    std::vector<base_iter> degree_ranges;
    deg_t max_degree;

public:
    using iterable = degree_range_iterator_helper<base_iter>;

    template <typename Vector>
    explicit graded_multiplication_helper(const Vector& rhs) : base_type(rhs)
    {
        ordering order;
        std::sort(right_buffer.begin(), right_buffer.end(), order);

        const auto& basis = rhs.basis();
        max_degree = basis_traits::max_degree(basis);
        degree_ranges.reserve(max_degree + 1);
        auto it = right_buffer.cbegin();
        auto end = right_buffer.cend();
        degree_ranges.push_back(it);
        auto degree = 0;

        for (; it != end; ++it) {
            auto current = basis_traits::degree(basis, it->first);
            for (; degree < current; ++degree) { degree_ranges.push_back(it); }
        }
        for (; degree <= max_degree; ++degree) { degree_ranges.push_back(end); }
    }

    iterable degree_range(deg_t degree) noexcept
    {
        if (degree < 0) {
            return iterable(right_buffer.begin(), right_buffer.begin());
        }
        if (degree > max_degree) {
            return iterable(right_buffer.end(), right_buffer.end());
        }
        return iterable(right_buffer.begin(), degree_ranges[degree + 1]);
    }
};

}// namespace dtl

template <
        typename Multiplier, typename Basis, dimn_t SSO = 1,
        typename Scalar = int>
class base_multiplier
{
    using basis_traits = basis_trait<Basis>;

public:
    using key_type = typename basis_traits::key_type;
    using scalar_type = Scalar;
    using pair_type = pair<key_type, scalar_type>;

    using product_type = boost::container::small_vector<pair_type, SSO>;
    using reference = const boost::container::small_vector_base<pair_type>&;

    static product_type uminus(reference arg)
    {
        product_type result;
        result.reserve(arg.size());
        for (const auto& item : arg) {
            result.emplace_back(item.first, -item.second);
        }
        return result;
    }

    static product_type add(reference lhs, reference rhs)
    {
        std::map<key_type, scalar_type> tmp;
        tmp.insert(lhs.begin(), lhs.end());

        for (const auto& item : rhs) { tmp[item.first] += item.second; }

        return {tmp.begin(), tmp.end()};
    }

    static product_type sub(reference lhs, reference rhs)
    {
        std::map<key_type, scalar_type> tmp;
        tmp.insert(lhs.begin(), lhs.end());

        for (const auto& item : rhs) { tmp[item.first] -= item.second; }

        return {tmp.begin(), tmp.end()};
    }

    product_type mul(const Basis& basis, reference lhs, key_type rhs) const
    {
        std::map<key_type, scalar_type> tmp;

        const auto& mult = static_cast<const Multiplier&>(*this);
        for (const auto& outer : lhs) {
            for (const auto& inner : mult(basis, outer.first, rhs)) {
                tmp[inner.first] += outer.second * inner.second;
            }
        }

        return {tmp.begin(), tmp.end()};
    }

    product_type mul(const Basis& basis, key_type lhs, reference rhs) const
    {
        std::map<key_type, scalar_type> tmp;

        const auto& mult = static_cast<const Multiplier&>(*this);
        for (const auto& outer : rhs) {
            for (const auto& inner : mult(basis, lhs, outer.first)) {
                tmp[inner.first] += inner.second * outer.second;
            }
        }

        return {tmp.begin(), tmp.end()};
    }

    product_type mul(const Basis& basis, reference lhs, reference rhs) const
    {
        std::map<key_type, scalar_type> tmp;

        const auto& mult = static_cast<const Multiplier&>(*this);
        for (const auto& litem : lhs) {
            for (const auto& ritem : rhs) {
                for (const auto& inner :
                     mult(basis, litem.first, ritem.first)) {
                    tmp[inner.first]
                            += inner.second * litem.second * ritem.second;
                }
            }
        }

        return {tmp.begin(), tmp.end()};
    }
};

namespace dtl {

template <typename Derived>
struct derived_or_this {
    template <typename Base>
    static const Derived& cast(const Base& arg) noexcept
    {
        return static_cast<const Derived&>(arg);
    }
};

template <>
struct derived_or_this<void> {
    template <typename Base>
    static const Base& cast(const Base& arg) noexcept
    {
        return arg;
    }
};

}// namespace dtl

template <typename Multiplier, typename Derived = void>
class base_multiplication
{

    template <typename V>
    using basis_t = typename V::basis_type;

    template <typename V>
    using key_tp = typename basis_trait<basis_t<V>>::key_type;

    template <typename V>
    using scal_t = typename V::scalar_type;

    template <typename V, typename Sca>
    using key_vect = std::vector<
            pair<typename basis_trait<typename V::basis_type>::key_type, Sca>>;

    template <typename V>
    using helper_type = dtl::general_multiplication_helper<
            typename V::basis_type, typename V::coefficient_ring>;
    template <typename V>
    using graded_helper_type = dtl::graded_multiplication_helper<
            typename V::basis_type, typename V::coefficient_ring>;

    //    template <typename OutVector, typename KeyProd, typename Sca>
    //    void asp_helper(OutVector& out, KeyProd&& key_prod, Sca&& scalar)
    //    const
    //    {
    //        using scalar_type = scal_t<OutVector>;
    //        scalar_type s(std::forward<Sca>(scalar));
    //        out.inplace_binary_op(std::forward<KeyProd>(key_prod), [s](scalar_type&
    //        lhs, const scalar_type& rhs) {
    //            return lhs += (rhs*s);
    //        });
    //    }

    using caster = dtl::derived_or_this<Derived>;

    template <typename OutVector, typename ProductType, typename PSca>
    void asp_helper(OutVector& out, ProductType&& key_prod, PSca&& scalar) const
    {
        using scalar_type = scal_t<OutVector>;
        scalar_type s(std::forward<PSca>(scalar));

        for (const auto& kv_pair : key_prod) {
            out[kv_pair.first] += scalar_type(kv_pair.second) * s;
        }
    }

protected:
    Multiplier m_mult;

public:
    using basis_type = typename Multiplier::basis_type;
    using key_type = typename basis_trait<basis_type>::key_type;
    using generic_ref
            = const boost::container::small_vector_base<pair<key_type, int>>&;

    template <
            typename... Args,
            typename
            = enable_if_t<is_constructible<Multiplier, Args...>::value>>
    explicit base_multiplication(Args&&... args)
        : m_mult(std::forward<Args>(args)...)
    {}

    template <typename Basis, typename Key>
    decltype(auto) multiply(const Basis& basis, Key lhs, Key rhs) const
    {
        return m_mult(basis, lhs, rhs);
    }

    template <typename Basis>
    decltype(auto)
    multiply_generic(const Basis& basis, generic_ref lhs, generic_ref rhs) const
    {
        return m_mult.mul(basis, lhs, rhs);
    }

    template <
            typename OutVector, typename LeftVector, typename RightVector,
            typename Fn>
    void
    fma(OutVector& out, const LeftVector& lhs, const RightVector& rhs,
        Fn fn) const
    {
        // The helper makes a contiguous copy of rhs key-value pairs
        // so the inner loop is always contiguous, rather than potentially
        // a linked list or other cache unfriendly data structure.
        helper_type<RightVector> helper(rhs);
        const auto& basis = out.basis();

        for (auto litem : lhs) {
            for (auto ritem : helper) {
                asp_helper(
                        out, m_mult(basis, litem.key(), ritem.first),
                        fn(litem.value() * ritem.second)
                );
            }
        }
    }

    template <
            typename OutVector, typename LeftVector, typename RightVector,
            typename Fn>
    void
    fma(OutVector& out, const LeftVector& lhs, const RightVector& rhs, Fn fn,
        deg_t max_deg) const
    {
        using out_basis_traits = basis_trait<typename OutVector::basis_type>;
        using lhs_basis_traits = basis_trait<typename LeftVector::basis_type>;

        // The helper makes a contiguous copy of rhs key-value pairs
        // so the inner loop is always contiguous, rather than potentially
        // a linked list or other cache unfriendly data structure.
        // The graded helper also sorts the keys and constructs a buffer
        // of degree ranges that we can use to truncate products that
        // would overflow the max degree.
        graded_helper_type<RightVector> helper(rhs);
        const auto& basis = out.basis();

        auto out_deg = std::min(
                out_basis_traits::max_degree(basis), lhs.degree() + rhs.degree()
        );
        out.update_degree(out_deg);
        const auto& lhs_basis = lhs.basis();
        for (auto litem : lhs) {
            auto lkey = litem.key();
            auto lhs_degree = lhs_basis_traits::degree(lhs_basis, lkey);
            auto rhs_degree = out_deg - lhs_degree;
            for (auto ritem : helper.degree_range(rhs_degree)) {
                asp_helper(
                        out, m_mult(basis, lkey, ritem.first),
                        fn(litem.value() * ritem.second)
                );
            }
        }
    }

    template <typename LeftVector, typename RightVector, typename Fn>
    void
    multiply_inplace(LeftVector& left, const RightVector& right, Fn op) const
    {
        if (!left.empty() && !right.empty()) {
            LeftVector tmp(left.get_basis());
            caster::cast(*this).fma(tmp, left, right, op);
            left = std::move(tmp);
        } else {
            left.clear();
        }
    }
    template <typename LeftVector, typename RightVector, typename Fn>
    void multiply_inplace(
            LeftVector& left, const RightVector& right, Fn op, deg_t max_deg
    ) const
    {
        if (!left.empty() && !right.empty()) {
            LeftVector tmp(left.get_basis());
            tmp.update_degree(std::min(left.degree() + right.degree(), max_deg)
            );
            caster::cast(*this).fma(tmp, left, right, op, max_deg);
            left = std::move(tmp);
        } else {
            left.clear();
        }
    }
};

template <
        typename Basis, typename Coefficients, typename Multiplication,
        template <typename, typename> class VectorType,
        template <typename> class StorageModel,
        template <
                typename, typename, template <typename, typename> class,
                template <typename> class, typename...>
        class Base
        = vector,
        typename... BaseArgs>
class algebra
    : public Base<Basis, Coefficients, VectorType, StorageModel, BaseArgs...>
{
    using base_type
            = Base<Basis, Coefficients, VectorType, StorageModel, BaseArgs...>;

public:
    using vector_type = vector<Basis, Coefficients, VectorType, StorageModel>;
    static_assert(
            is_same<base_type, vector_type>::value
                    || is_base_of<vector_type, base_type>::value,
            "algebra must derive from vector"
    );

    using multiplication_type = Multiplication;

    using typename vector_type::basis_pointer;
    using multiplication_pointer = std::shared_ptr<const multiplication_type>;

    using typename vector_type::basis_type;
    using typename vector_type::coefficient_ring;
    using typename vector_type::key_type;
    using typename vector_type::rational_type;
    using typename vector_type::scalar_type;

private:
    multiplication_pointer p_mult;

public:
    //    using vector_type::vector_type;

    algebra()
        : vector_type(), p_mult(multiplication_registry<Multiplication>::get(
                                 vector_type::basis()
                         ))
    {}

    explicit algebra(basis_pointer basis)
        : vector_type(std::move(basis)),
          p_mult(multiplication_registry<Multiplication>::get(
                  vector_type::basis()
          ))
    {}

    explicit algebra(vector_type&& arg)
        : vector_type(std::move(arg)),
          p_mult(multiplication_registry<Multiplication>::get(
                  vector_type::basis()
          ))
    {}

    template <typename Scalar>
    algebra(basis_pointer basis, std::shared_ptr<const Multiplication> mul,
            std::initializer_list<Scalar> args)
        : vector_type(std::move(basis), args), p_mult(std::move(mul))
    {}

    algebra(basis_pointer basis, std::shared_ptr<const Multiplication> mult)
        : vector_type(basis), p_mult(std::move(mult))
    {}

    algebra(const vector_type& base, std::shared_ptr<const Multiplication> mult)
        : vector_type(base), p_mult(std::move(mult))
    {}

    template <
            typename... Args,
            typename
            = enable_if_t<is_constructible<vector_type, Args...>::value>>
    explicit algebra(basis_pointer basis, Args... args)
        : vector_type(std::move(basis), std::forward<Args>(args)...),
          p_mult(multiplication_registry<Multiplication>::get(
                  vector_type::basis()
          ))
    {}

    template <
            typename... Args,
            typename
            = enable_if_t<is_constructible<vector_type, Args...>::value>>
    explicit algebra(
            basis_pointer basis, std::shared_ptr<const Multiplication> mul,
            Args&&... args
    )
        : vector_type(std::move(basis), std::forward<Args>(args)...), p_mult(std::move(mul))
    {}

    template <
            template <typename, typename> class OtherVectorType,
            template <typename> class OtherStorageModel>
    explicit algebra(const algebra<
                     Basis, Coefficients, Multiplication, OtherVectorType,
                     OtherStorageModel>& other)
        : vector_type((other)), p_mult(other.p_mult)
    {}

    algebra(const algebra& other) : vector_type(other), p_mult(other.p_mult) {}

    algebra(algebra&& other) noexcept
        : vector_type(static_cast<vector_type&&>(other)),
          p_mult(std::move(other.p_mult))
    {}

    algebra& operator=(const algebra& other)
    {
        if (&other != this) {
            vector_type::operator=(other);
            p_mult = other.p_mult;
        }
        return *this;
    }

    algebra& operator=(algebra&& other) noexcept
    {
        if (&other != this) {
            p_mult = std::move(other.p_mult);
            vector_type::operator=(static_cast<vector_type&&>(other));
        }
        return *this;
    }

    std::shared_ptr<const multiplication_type> multiplication() const noexcept
    {
        return p_mult;
    }

    algebra create_alike() const {
        return algebra(this->get_basis(), p_mult);
    }

    algebra& add_mul(const algebra& lhs, const algebra& rhs)
    {
        using traits = multiplication_traits<Multiplication>;
        traits::multiply_and_add(
                *multiplication(), *this, lhs, rhs,
                [](scalar_type s) { return s; }
        );
        return *this;
    }
    algebra& sub_mul(const algebra& lhs, const algebra& rhs)
    {
        using traits = multiplication_traits<Multiplication>;
        traits::multiply_and_add(
                *multiplication(), *this, lhs, rhs,
                [](scalar_type s) { return -s; }
        );
        return *this;
    }

    algebra& mul_scal_prod(const algebra& rhs, const scalar_type& scal)
    {
        using traits = multiplication_traits<Multiplication>;
        traits::multiply_inplace(
                *multiplication(), *this, rhs,
                [scal](scalar_type s) { return s * scal; }
        );
        return *this;
    }
    algebra& mul_scal_div(const algebra& rhs, const rational_type& scal)
    {
        using traits = multiplication_traits<Multiplication>;
        traits::multiply_inplace(
                *multiplication(), *this, rhs,
                [scal](scalar_type s) { return s / scal; }
        );
        return *this;
    }

    algebra&
    mul_scal_div(const algebra& rhs, const rational_type& scal, deg_t degree)
    {
        using traits = multiplication_traits<Multiplication>;
        traits::multiply_inplace(
                *multiplication(), *this, rhs,
                [scal](scalar_type s) { return s / scal; }, degree
        );
        return *this;
    }
};

template <
        typename Basis, typename Coefficients, typename Multiplication,
        template <typename, typename> class VectorType,
        template <typename> class StorageModel,
        typename Base = algebra<
                Basis, Coefficients, Multiplication, VectorType, StorageModel>>
class unital_algebra : public Base
{
    using algebra_base = algebra<
            Basis, Coefficients, Multiplication, VectorType, StorageModel>;

    static_assert(
            is_same<Base, algebra_base>::value
                    || is_base_of<algebra_base, Base>::value,
            "algebra types must derive from algebra"
    );

public:
    using typename algebra_base::basis_pointer;
    using typename algebra_base::scalar_type;

    using Base::Base;

    template <
            typename Scalar,
            typename
            = enable_if_t<is_constructible<scalar_type, Scalar>::value>>
    explicit unital_algebra(basis_pointer basis, Scalar val)
        : algebra_base(
                std::move(basis),
                basis_trait<Basis>::first_key(*algebra_base::p_basis),
                scalar_type(val)
        )
    {}

    template <
            typename Scalar,
            typename
            = enable_if_t<is_constructible<scalar_type, Scalar>::value>>
    explicit unital_algebra(Scalar val)
        : algebra_base(
                basis_trait<Basis>::first_key(*algebra_base::p_basis),
                scalar_type(val)
        )
    {}
};

template <
        typename Basis, typename Coefficients, typename Multiplication,
        template <typename, typename> class VectorType,
        template <typename> class StorageModel,
        typename Base = algebra<
                Basis, Coefficients, Multiplication, VectorType, StorageModel>>
class graded_algebra : public Base
{

    using algebra_base = algebra<
            Basis, Coefficients, Multiplication, VectorType, StorageModel>;

    static_assert(
            is_same<Base, algebra_base>::value
                    || is_base_of<algebra_base, Base>::value,
            "algebra types must derive from algebra"
    );

public:
    using Base::Base;
};

template <
        typename Basis, typename Coefficients, typename Multiplication,
        template <typename, typename> class VectorType,
        template <typename> class StorageModel>
using graded_unital_algebra = unital_algebra<
        Basis, Coefficients, Multiplication, VectorType, StorageModel,
        graded_algebra<
                Basis, Coefficients, Multiplication, VectorType, StorageModel>>;

namespace dtl {

template <typename Algebra>
class is_algebra
{
    template <
            typename Basis, typename Coefficients, typename Multiplication,
            template <typename, typename> class VType,
            template <typename> class SModel>
    static true_type
    test(algebra<Basis, Coefficients, Multiplication, VType, SModel>&);

    static false_type test(...);

public:
    static constexpr bool value = decltype(test(declval<Algebra&>()))::value;
};

}// namespace dtl

template <typename Algebra>
enable_if_t<dtl::is_algebra<Algebra>::value, Algebra>
operator*(const Algebra& lhs, const Algebra& rhs)
{
    using traits = multiplication_traits<typename Algebra::multiplication_type>;
    using scalar_type = typename Algebra::scalar_type;

    auto multiplication = lhs.multiplication();
    if (!multiplication) { multiplication = rhs.multiplication(); }
    Algebra result(lhs.get_basis(), multiplication);
    if (multiplication && !lhs.empty() && !rhs.empty()) {
        traits::multiply_and_add(
                *multiplication, result, lhs, rhs,
                [](scalar_type s) { return s; }
        );
    }
    return result;
}

template <typename Algebra>
enable_if_t<dtl::is_algebra<Algebra>::value, Algebra&>
operator*=(Algebra& lhs, const Algebra& rhs)
{
    using traits = multiplication_traits<typename Algebra::multiplication_type>;
    using scalar_type = typename Algebra::scalar_type;
    if (rhs.empty()) {
        lhs.clear();
        return lhs;
    }
    auto multiplication = lhs.multiplication();
    if (!multiplication) { multiplication = rhs.multiplication(); }

    if (multiplication && !lhs.empty()) {
        traits::multiply_inplace(
                *multiplication, lhs, rhs, [](scalar_type arg) { return arg; }
        );
    }
    return lhs;
}

template <typename Algebra>
enable_if_t<dtl::is_algebra<Algebra>::value, Algebra>
commutator(const Algebra& lhs, const Algebra& rhs)
{
    using traits = multiplication_traits<typename Algebra::multiplication_type>;
    using scalar_type = typename Algebra::scalar_type;

    auto multiplication = lhs.multiplication();
    if (!multiplication) { multiplication = rhs.multiplication(); }

    Algebra result(lhs.get_basis(), multiplication);
    if (multiplication && !lhs.empty() && !rhs.empty()) {
        traits::multiply_and_add(*multiplication, result, lhs, rhs);
        traits::multiply_and_add(
                *multiplication, result, rhs, lhs, [](const scalar_type& arg) {
                    return -arg;
                }
        );
    }
    return result;
}



template <typename Multiplication, typename LVector, typename RVector>
LVector multiply(const Multiplication& multiplication, const LVector& left,
                 const
                 RVector& right) {
    using traits = multiplication_traits<Multiplication>;
    LVector result = left.create_alike();
    traits::multiply_and_add(multiplication, result, left, right);
    return result;
}


}// namespace lal

#endif// LIBALGEBRA_LITE_ALGEBRA_H
