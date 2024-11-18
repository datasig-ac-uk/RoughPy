//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_NUMBER_TRAIT_H
#define ROUGHPY_GENERICS_NUMBER_TRAIT_H

#include <cmath>

#include <roughpy/core/debug_assertion.h>
#include "roughpy/core/hash.h"

#include "builtin_trait.h"
#include "type_ptr.h"

namespace rpy::generics {

class ConstRef;
class Ref;
class Value;


class ROUGHPY_PLATFORM_EXPORT NumberTrait : public BuiltinTrait
{
    const Type* p_type;
    const Type* p_real_type;

protected:

    explicit NumberTrait(const Type* type, const Type* real_type=nullptr)
        : BuiltinTrait(my_id),
          p_type(type),
          p_real_type(real_type)
    {}

public:

    static constexpr auto my_id = BuiltinTraitID::Number;

    virtual ~NumberTrait();

    virtual bool has_function(NumberFunction fn_id) const noexcept = 0;

    virtual void unsafe_real(void* dst, const void* src) const;
    virtual void unsafe_imaginary(void* dst, const void* src) const;

    virtual void unsafe_abs(void* dst, const void* src) const noexcept = 0;


    virtual void unsafe_sqrt(void* dst, const void* src) const;
    virtual void unsafe_pow(void* dst, const void* base, exponent_t exponent) const;

    virtual void unsafe_exp(void* dst, const void* src) const;
    virtual void unsafe_log(void* dst, const void* src) const;

    virtual void unsafe_from_rational(void* dst, int64_t numerator,
        int64_t denominator) const = 0;


    void real(Ref dst, ConstRef src) const;
    void imaginary(Ref dst, ConstRef src) const;
    void abs(Ref dst, ConstRef src) const;
    void sqrt(Ref dst, ConstRef src) const;
    void exp(Ref dst, ConstRef src) const;
    void log(Ref dst, ConstRef src) const;

    void from_rational(Ref dst, int64_t numerator, int64_t denominator) const;

    RPY_NO_DISCARD Value real(ConstRef value) const;
    RPY_NO_DISCARD Value imaginary(ConstRef value) const;
    RPY_NO_DISCARD Value abs(ConstRef value) const;
    RPY_NO_DISCARD Value sqrt(ConstRef value) const;
    RPY_NO_DISCARD Value pow(ConstRef value, exponent_t power) const;
    RPY_NO_DISCARD Value exp(ConstRef value) const;
    RPY_NO_DISCARD Value log(ConstRef value) const;

    RPY_NO_DISCARD Value
    from_rational(int64_t numerator, int64_t denominator) const;

    RPY_NO_DISCARD
    const Type* real_type() const noexcept
    {
        return this->p_real_type ? this->p_real_type : this->p_type;
    }

    RPY_NO_DISCARD const Type* imaginary_type() const noexcept
    {
        return this->p_real_type;
    }
};




template <typename T>
class NumberTraitImpl : public NumberTrait
{

public:

    using real_type = T;

    explicit NumberTraitImpl(const Type* type, const Type* real_type = nullptr)
        : NumberTrait(type, real_type)
    {
        RPY_DBG_ASSERT_NE(type, nullptr);
    }

    bool has_function(NumberFunction fn_id) const noexcept override;

    void unsafe_real(void* dst, const void* src) const override;
    void unsafe_imaginary(void* dst, const void* src) const override;
    void unsafe_abs(void* dst, const void* src) const noexcept override;
    void unsafe_sqrt(void* dst, const void* src) const override;
    void
    unsafe_pow(void* dst, const void* base, exponent_t exponent) const override;
    void unsafe_exp(void* dst, const void* src) const override;
    void unsafe_log(void* dst, const void* src) const override;
    void unsafe_from_rational(void* dst, int64_t numerator, int64_t denominator)
            const override;
};

namespace number_trait_impl {

// Bring the whole std namespace into scope, so std::abs etc can participate in
// name resolution for standard types.
using namespace std;

struct NoSuchFunction {
    using result_t = byte;
    static constexpr bool has_function = false;

    template <typename... Args>
    RPY_NO_RETURN result_t operator()(Args&&... args) const
    {
        RPY_THROW(std::runtime_error, "function not defined");
    }
};

template <typename T, typename = void>
struct AbsFunc : NoSuchFunction {
};

template <typename T>
struct AbsFunc<T, void_t<decltype(abs(declval<T>()))>> {
    using result_t = decltype(abs(declval<T>()));

    static constexpr bool has_function = true;
    constexpr result_t operator()(const T& val) const
            noexcept(noexcept(abs(val)))
    {
        return abs(val);
    }
};

template <typename T, typename = void>
struct SqrtFunc : NoSuchFunction {
};

template <typename T>
struct SqrtFunc<T, void_t<decltype(sqrt(declval<T>()))>> {
    using result_t = decltype(sqrt(declval<T>()));

    static constexpr bool has_function = true;
    constexpr result_t operator()(const T& val) const
            noexcept(noexcept(sqrt(val)))
    {
        return sqrt(val);
    }
};

template <typename T, typename = void>
struct PowFunc : NoSuchFunction {
};

template <typename T>
struct PowFunc<
        T,
        void_t<decltype(sqrt(declval<T>()), std::declval<exponent_t>())>> {
    static constexpr bool has_function = true;
    using result_t
            = decltype(pow(std::declval<T>(), std::declval<exponent_t>()));

    constexpr result_t operator()(const T& val, exponent_t exponent) const
            noexcept(noexcept(pow(val, exponent)))
    {
        return pow(val, exponent);
    }
};

template <typename T, typename = void>
struct ExpFunc : NoSuchFunction {
};

template <typename T>
struct ExpFunc<T, void_t<decltype(exp(declval<T>()))>> {
    using result_t = decltype(exp(declval<T>()));
    static constexpr bool has_function = true;

    constexpr result_t operator()(const T& val) const
            noexcept(noexcept(exp(val)))
    {
        return exp(val);
    }
};

template <typename T, typename = void>
struct LogFunc : NoSuchFunction {
};

template <typename T>
struct LogFunc<T, void_t<decltype(log(declval<T>()))>> {
    using result_t = decltype(log(declval<T>()));

    static constexpr bool has_function = true;
    constexpr result_t operator()(const T& val) const
            noexcept(noexcept(log(val)))
    {
        return log(val);
    }
};

}// namespace number_trait_impl

template <typename T>
bool NumberTraitImpl<T>::has_function(NumberFunction fn_id) const noexcept
{
    switch (fn_id) {
        case NumberFunction::Abs:
            return number_trait_impl::AbsFunc<T>::has_function;
        case NumberFunction::Sqrt:
            return number_trait_impl::SqrtFunc<T>::has_function;
        case NumberFunction::Pow:
            return number_trait_impl::PowFunc<T>::has_function;
        case NumberFunction::Exp:
            return number_trait_impl::ExpFunc<T>::has_function;
        case NumberFunction::Log:
            return number_trait_impl::LogFunc<T>::has_function;
        case NumberFunction::FromRational:
            return is_floating_point_v<T>;
        case NumberFunction::Real:
        case NumberFunction::Imaginary:
            return true; // all the builtin types are real
    }
    RPY_UNREACHABLE_RETURN(false);
}

template <typename T>
void NumberTraitImpl<T>::unsafe_real(void* dst, const void* src) const
{
    NumberTrait::unsafe_real(dst, src);
}
template <typename T>
void NumberTraitImpl<T>::unsafe_imaginary(void* dst, const void* src) const
{
    NumberTrait::unsafe_imaginary(dst, src);
}
template <typename T>
void NumberTraitImpl<T>::unsafe_abs(void* dst, const void* src) const noexcept
{
    using Fn = number_trait_impl::AbsFunc<T>;
    Fn fn;
    *static_cast<real_type*>(dst) = fn(*static_cast<const T*>(src));
}
template <typename T>
void NumberTraitImpl<T>::unsafe_sqrt(void* dst, const void* src) const
{
    using Fn = number_trait_impl::SqrtFunc<T>;
    Fn fn;
    *static_cast<T*>(dst) = fn(*static_cast<const T*>(src));
}
template <typename T>
void NumberTraitImpl<T>::unsafe_pow(
        void* dst,
        const void* base,
        exponent_t exponent
) const
{
    using Fn = number_trait_impl::PowFunc<T>;
    Fn fn;
    auto result = fn(*static_cast<const T*>(base), exponent);
    *static_cast<T*>(dst) = result;
}
template <typename T>
void NumberTraitImpl<T>::unsafe_exp(void* dst, const void* src) const
{
    using Fn = number_trait_impl::ExpFunc<T>;
    Fn fn;
    auto result = fn(*static_cast<const T*>(src));
    *static_cast<T*>(dst) = result;
}
template <typename T>
void NumberTraitImpl<T>::unsafe_log(void* dst, const void* src) const
{
    using Fn = number_trait_impl::LogFunc<T>;
    Fn fn;
    *static_cast<T*>(dst) = fn(*static_cast<const T*>(src));
}
template <typename T>
void NumberTraitImpl<T>::unsafe_from_rational(
        void* dst,
        int64_t numerator,
        int64_t denominator
) const
{
    if constexpr (is_floating_point_v<T>) {
        *static_cast<T*>(dst) = static_cast<T>(numerator) / static_cast<T>(denominator);
    } else {
        RPY_THROW(std::domain_error, "cannot convert from rational");
    }
}

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_NUMBER_TRAIT_H
