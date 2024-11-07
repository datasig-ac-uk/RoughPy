//
// Created by user on 06/02/23.
//

#ifndef LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_OPERATORS_H
#define LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_OPERATORS_H

#include "implementation_types.h"

#include <utility>
#include <type_traits>
#include <tuple>

#include "vector.h"

namespace lal {

namespace dtl {
template <typename T> struct is_vector;
}

namespace operators {

#define LAL_LO_DECLRESULT(IMPL) \
    decltype(std::declval<IMPL>()(std::declval<const ArgumentType&>()))

namespace dtl {

template <typename Impl1, typename Impl2>
class sum_operator;


template <typename Impl, typename Multiplier>
class scalar_multiply_operator;

template <typename Scalar>
class left_scalar_multiply;

template <typename Scalar>
class right_scalar_multiply;


template <typename... Impl>
class composition_operator;

template <typename Operator>
struct operator_traits;


} // namespace dtl


template <typename Impl>
class linear_operator : protected Impl
{
    using traits = dtl::operator_traits<Impl>;
public:
    using scalar_type = typename traits::scalar_type;

protected:
    using implementation_type = Impl;

public:

    template <typename... Args>
    explicit linear_operator(Args&&... args) : Impl(std::forward<Args>(args)...)
    {}

    using implementation_type::operator();


    template <typename Impl2>
    friend linear_operator<dtl::sum_operator<Impl, Impl2>>
    operator+(const linear_operator& left,
              const linear_operator<Impl2>& right)
    {
        // Sum passes on the implementation to the sum operator
        return { static_cast<const Impl&>(left), static_cast<const Impl2&>(right) };
    }

    template <typename Scalar>
    friend linear_operator<dtl::scalar_multiply_operator<Impl, dtl::left_scalar_multiply<scalar_type>>>
    operator*(const Scalar& scalar, const linear_operator& op)
    {
        return { static_cast<const Impl&>(op), scalar_type(scalar) };
    }

    template <typename Scalar>
    friend linear_operator<dtl::scalar_multiply_operator<Impl, dtl::right_scalar_multiply<scalar_type>>>
    operator*(const linear_operator& op, const Scalar& scalar)
    {
        return { static_cast<const Impl&>(op), scalar_type(scalar) };
    }


    template <typename... Impls>
    std::enable_if_t<(sizeof...(Impls) > 1), linear_operator<dtl::composition_operator<Impls...>>>
    compose(const linear_operator<Impls>&... operators)
    {
        return { static_cast<Impls>(operators)... };
    }

};




namespace dtl {

template <typename Impl1, typename Impl2>
class sum_operator
{
    Impl1 m_left;
    Impl2 m_right;

    template <typename LeftType, typename RightType>
    static LeftType add_results(LeftType&& left, RightType&& right) {
        for (auto rit : right) {
            left.add_scal_prod(rit.key(), rit.value());
        }
        return left;
    }

    template <typename Type>
    static Type add_results(Type&& left, Type&& right) {
        left += right;
        return left;
    }

public:

    sum_operator(const Impl1& left, const Impl2& right)
        : m_left(left), m_right(right)
    {}

    sum_operator(Impl1&& left, Impl2&& right)
        : m_left(std::move(left)), m_right(std::move(right))
    {}

    template <typename Argument>
    auto operator()(const Argument &arg) const -> decltype(add_results(m_left(arg), m_right(arg)))
    {
        return add_results(m_left(arg), m_right(arg));
    }

};

template <typename Scalar>
class left_scalar_multiply {
    Scalar m_scalar;

public:

    using scalar_type = Scalar;

    explicit left_scalar_multiply(Scalar&& scal) : m_scalar(std::move(scal))
    {}

    template <typename Vector>
    Vector multiply(Vector&& arg) const
    {
        return m_scalar * arg;
    }

};


template <typename Scalar>
class right_scalar_multiply {
    Scalar m_scalar;

public:

    using scalar_type = Scalar;

    explicit right_scalar_multiply(Scalar&& scal) : m_scalar(std::move(scal))
    {}

    template <typename Vector>
    Vector multiply(Vector&& arg) const
    {
        arg *= m_scalar;
        return arg;
    }

};


template <typename Impl, typename Multiplier>
class scalar_multiply_operator : private Multiplier
{
    Impl m_operator;

public:

    explicit scalar_multiply_operator(const Impl& op, const typename Multiplier::scalar_type& s)
        : Multiplier(s), m_operator(op)
    {}

    explicit scalar_multiply_operator(Impl&& op, typename Multiplier::scalar_type&& s)
        : Multiplier(std::move(s)), m_operator(std::move(op))
    {}

    template <typename Argument>
    auto operator()(const Argument& arg) const -> decltype(m_operator(arg))
    {
        return Multiplier::multiply(m_operator(arg));
    }

};

template <std::size_t I, typename Argument, typename... Impl>
static constexpr auto eval_recursive(const Argument &arg, const std::tuple<Impl...>& ops)
        -> decltype(std::get<I>(ops)(arg)) {
    return eval_recursive<I+1>(std::get<I>(ops)(arg), ops);
}


template <typename Argument, typename... Impl>
static constexpr auto eval_recursive<sizeof...(Impl)-1, Argument, Impl...>(
    const Argument &arg, const std::tuple<Impl...>& ops)
        -> decltype(std::get<sizeof...(Impl)-1>(ops)(arg)) {
    return std::get<sizeof...(Impl)-1>(ops)(arg), ops);
}



template <typename... Impl>
class composition_operator
{
    static_assert(sizeof...(Impl) > 1, "composition of a single operator is not allowed");

    using operator_tuple = std::tuple<Impl...>;
    std::tuple<Impl...> m_operators;

    template <std::size_t I>
    struct eval_recursive
    {
        using next = eval_recursive<I+1>;

        template <typename Arg>
        constexpr auto eval(const Arg& arg, const operator_tuple& ops)
            -> decltype(std::get<I>(ops)(next::eval(arg, ops)))
        {
            return std::get<I>(ops)(next::eval(arg, ops));
        }
    };

    template <>
    struct eval_recursive<sizeof...(Impl)>
    {
        template <typename Arg>
        constexpr const Arg& eval(const Arg& arg, const operator_tuple& ops)
        {
            return arg;
        }
    };

    using evaluator = eval_recursive<0>;


public:

    composition_operator(const Impl&... impls) : m_operators(impls...)
    {}

    composition_operator(Impl&&... impls) : m_operators(std::forward<Impl>(impls)...)
    {}

    template <typename Argument>
    auto operator()(const Argument& arg) const -> decltype(evaluator::eval(arg, m_operators))
    {
        return evaluator::eval(arg, m_operators);
    }

};




} // namespace dtl


#undef LAL_LO_DECLRESULT





} // namespace operators
} // namespace lal


#endif //LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_OPERATORS_H
