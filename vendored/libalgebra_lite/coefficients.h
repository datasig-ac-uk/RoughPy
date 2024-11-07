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

#ifndef LIBALGEBRA_LITE_COEFFICIENTS_H
#define LIBALGEBRA_LITE_COEFFICIENTS_H

#include "implementation_types.h"
#include "libalgebra_lite_export.h"

#include <memory>

#include "detail/traits.h"

#ifdef LAL_ENABLE_RATIONAL_COEFFS
#include "rationals.h"
#endif

namespace lal {

template <typename Coeff>
struct coefficient_trait {
    using coefficient_ring = Coeff;
    using scalar_type = typename Coeff::scalar_type;
    using rational_type = typename Coeff::rational_type;
    using default_alloc = std::allocator<scalar_type>;
};

#define LAL_RING_GENERATE_BINOP(NAME, OP, RET_T, LHS_T, RHS_T)                 \
    template <typename Lhs = LHS_T, typename Rhs = RHS_T>                      \
    static constexpr RET_T NAME(const Lhs& lhs, const Rhs& rhs)                \
    {                                                                          \
        return lhs OP rhs;                                                     \
    }                                                                          \
                                                                               \
    template <typename Rhs = RHS_T>                                            \
    static constexpr RET_T NAME##_inplace(RET_T& lhs, const Rhs& rhs)          \
    {                                                                          \
        return (lhs OP## = rhs);                                               \
    }

template <typename Scalar, typename Rational>
struct coefficient_ring {
    using scalar_type = Scalar;
    using rational_type = Rational;

    static const scalar_type& zero() noexcept
    {
        static const scalar_type zero{};
        return zero;
    }
    static const scalar_type& one() noexcept
    {
        static const scalar_type one(1);
        return one;
    }
    static const scalar_type& mone() noexcept
    {
        static const scalar_type mone(-1);
        return mone;
    }

    static constexpr scalar_type uminus(const scalar_type& arg) { return -arg; }

    static inline bool is_invertible(const scalar_type& arg) noexcept
    {
        return arg != zero();
    }
    static constexpr const rational_type& as_rational(const scalar_type& arg
    ) noexcept
    {
        static_assert(
                is_convertible<const Scalar&, const Rational&>::value,
                "default conversion to rational is only defined when scalar "
                "type "
                "is convertible to rational type (as references)"
        );
        return static_cast<const rational_type&>(arg);
    }

    LAL_RING_GENERATE_BINOP(add, +, scalar_type, scalar_type, scalar_type)
    LAL_RING_GENERATE_BINOP(sub, -, scalar_type, scalar_type, scalar_type)
    LAL_RING_GENERATE_BINOP(mul, *, scalar_type, scalar_type, scalar_type)
    LAL_RING_GENERATE_BINOP(div, /, scalar_type, scalar_type, rational_type)
};
#undef LAL_RING_GENERATE_BINOP

template <typename Scalar>
struct coefficient_field : public coefficient_ring<Scalar, Scalar> {
};

LAL_EXPORT_TEMPLATE_STRUCT(coefficient_field, double)
LAL_EXPORT_TEMPLATE_STRUCT(coefficient_field, float)

using double_field = coefficient_field<double>;
using float_field = coefficient_field<float>;

#ifdef LAL_ENABLE_RATIONAL_COEFFS
LAL_EXPORT_TEMPLATE_STRUCT(coefficient_field, dtl::rational_scalar_type)
using rational_field = coefficient_field<dtl::rational_scalar_type>;
#endif

template <>
struct coefficient_trait<float> {
    using coefficient_ring = float_field;
    using scalar_type = float;
    using rational_type = float;
};

template <>
struct coefficient_trait<double> {
    using coefficient_ring = double_field;
    using scalar_type = double;
    using rational_type = double;
};

#ifdef LAL_ENABLE_RATIONAL_COEFFS
template <>
struct coefficient_trait<dtl::rational_scalar_type> {
    using coefficient_ring = rational_field;
    using scalar_type = dtl::rational_scalar_type;
    using rational_type = dtl::rational_scalar_type;
};
#endif

namespace ops {

template <typename Scalar>
struct identity {
//    template <typename S>
//    enable_if_t<is_same<Scalar, S>::value, const Scalar&>
//    operator()(const S& arg) const
//    {
//        return arg;
//    }
//    template <typename S>
//    enable_if_t<!is_same<Scalar, S>::value, Scalar>
//    operator()(const S& arg) const
//    {
//        return static_cast<Scalar>(arg);
//    }
    Scalar operator()(Scalar arg) const { return arg; }
};

template <typename Scalar>
struct unary_minus {
    template <typename S>
    Scalar operator()(const S& arg) const
    {
        return -static_cast<Scalar>(arg);
    }
};

template <typename Scalar>
struct add {
    template <typename L, typename R = L>
    Scalar operator()(const L& left, const R& right) const
    {
        return static_cast<Scalar>(left) + right;
    }
};

template <typename Scalar>
struct sub {
    template <typename L, typename R = L>
    Scalar operator()(const L& left, const R& right) const
    {
        return static_cast<Scalar>(left) - right;
    }
};

template <typename Scalar>
struct multiply {
    template <typename L, typename R = L>
    Scalar operator()(const L& left, const R& right) const
    {
        return static_cast<Scalar>(left) * right;
    }
};

template <typename Scalar>
struct divide {
    template <typename L, typename R = L>
    Scalar operator()(const L& left, const R& right) const
    {
        return static_cast<Scalar>(left) / right;
    }
};

struct add_inplace {
//    template <typename S, typename R = S>
//    S& operator()(S& left, const R& right) const
//    {
//        return left += right;
//    }

    template <typename S, typename R>
    S operator()(S left, const R& right) const
    {
        return left += right;
    }
};

struct sub_inplace {
    template <typename S, typename R = S>
    S operator()(S left, const R& right) const
    {
        return left -= right;
    }
};

struct multiply_inplace {
    template <typename S, typename R = S>
    S operator()(S left, const R& right) const
    {
        return left *= right;
    }
};

struct divide_inplace {
    template <typename S, typename R = S>
    S operator()(S left, const R& right) const
    {
        return left /= right;
    }
};

template <typename Scalar, typename M=Scalar>
struct pre_multiply {
    M multiplier;

    template <
            typename _M,
            typename = enable_if_t<!is_same<_M, pre_multiply>::value>>
    explicit pre_multiply(_M&& arg) : multiplier(std::forward<_M>(arg))
    {}

    template <typename S=M>
    Scalar operator()(const S& arg) const
    {
        return multiplier * static_cast<Scalar>(arg);
    }
};

template <typename Scalar, typename M=Scalar>
struct post_multiply {
    M multiplier;

    template <
            typename _M,
            typename = enable_if_t<!is_same<_M, post_multiply>::value>>
    explicit post_multiply(_M&& arg) : multiplier(std::forward<_M>(arg))
    {}

    template <typename S=M>
    Scalar operator()(const S& arg) const
    {
        return static_cast<Scalar>(arg) * multiplier;
    }
};

template <typename Scalar, typename D>
struct post_divide {
    D divisor;

    template <
            typename _D,
            typename = enable_if_t<!is_same<_D, post_divide>::value>>
    explicit post_divide(_D&& arg) : divisor(std::forward<_D>(arg))
    {}

    template <typename S>
    Scalar operator()(const S& arg) const
    {
        return static_cast<Scalar>(arg) / divisor;
    }
};

template <typename M>
struct post_multiply_inplace {
    M multiplier;

    template <
            typename _M,
            typename = enable_if_t<!is_same<_M, post_multiply_inplace>::value>>
    explicit post_multiply_inplace(_M&& arg) : multiplier(std::forward<_M>(arg))
    {}

    template <typename S>
    S& operator()(S& arg) const
    {
        return arg *= multiplier;
    }
};

template <typename D>
struct post_divide_inplace {
    D divisor;

    template <
            typename _D,
            typename = enable_if_t<!is_same<_D, post_divide_inplace>::value>>
    explicit post_divide_inplace(_D&& arg) : divisor(std::forward<_D>(arg))
    {}

    template <typename S>
    S& operator()(S& arg) const
    {
        return arg /= divisor;
    }
};

template <typename M>
struct add_pre_multiply_inplace {
    M multiplier;

    template <
            typename _M,
            typename
            = enable_if_t<!is_same<_M, add_pre_multiply_inplace>::value>>
    explicit add_pre_multiply_inplace(_M&& m) : multiplier(std::forward<_M>(m))
    {}

    template <typename S, typename R = S>
    S& operator()(S& left, const R& right) const
    {
        return left += multiplier * right;
    }
};
template <typename M>
struct add_post_multiply_inplace {
    M multiplier;

    template <
            typename _M,
            typename
            = enable_if_t<!is_same<_M, add_post_multiply_inplace>::value>>
    explicit add_post_multiply_inplace(_M&& m) : multiplier(std::forward<_M>(m))
    {}

    template <typename S, typename R = S>
    S operator()(S left, const R& right) const
    {
        return left += right * multiplier;
    }
};

template <typename D>
struct add_post_divide_inplace {
    D divisor;

    template <
            typename _D,
            typename
            = enable_if_t<!is_same<_D, add_post_divide_inplace>::value>>
    explicit add_post_divide_inplace(_D&& d) : divisor(std::forward<_D>(d))
    {}

    template <typename S, typename R = S>
    S operator()(S left, const R& right) const
    {
        return left += right / divisor;
    }
};

template <typename M>
struct sub_pre_multiply_inplace {
    M multiplier;

    template <
            typename _M,
            typename
            = enable_if_t<!is_same<_M, sub_pre_multiply_inplace>::value>>
    explicit sub_pre_multiply_inplace(_M&& m) : multiplier(std::forward<_M>(m))
    {}

    template <typename S, typename R = S>
    S operator()(S left, const R& right) const
    {
        return left -= multiplier * right;
    }
};
template <typename M>
struct sub_post_multiply_inplace {
    M multiplier;

    template <
            typename _M,
            typename
            = enable_if_t<!is_same<_M, sub_post_multiply_inplace>::value>>
    explicit sub_post_multiply_inplace(_M&& m) : multiplier(std::forward<_M>(m))
    {}

    template <typename S, typename R = S>
    S operator()(S left, const R& right) const
    {
        return left -= right * multiplier;
    }
};

template <typename D>
struct sub_post_divide_inplace {
    D divisor;

    template <
            typename _D,
            typename
            = enable_if_t<!is_same<_D, sub_post_divide_inplace>::value>>
    explicit sub_post_divide_inplace(_D&& d) : divisor(std::forward<_D>(d))
    {}

    template <typename S, typename R = S>
    S operator()(S left, const R& right) const
    {
        return left -= right / divisor;
    }
};

}// namespace ops

}// namespace lal

#endif// LIBALGEBRA_LITE_COEFFICIENTS_H
