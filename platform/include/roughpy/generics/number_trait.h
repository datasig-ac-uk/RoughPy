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

    using exponent_t = int;
    static constexpr auto my_id = BuiltinTraitID::Number;



    virtual ~NumberTrait();

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
class ROUGHPY_PLATFORM_NO_EXPORT NumberTraitImpl : public NumberTrait
{

public:
    explicit NumberTraitImpl(const Type* type, const Type* real_type = nullptr)
        : NumberTrait(type, real_type)
    {
        RPY_DBG_ASSERT_NE(type, nullptr);
    }

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

struct NoSuchFunction
{

    using result_t = byte;

    template <typename... Args>
    RPY_NO_RETURN
    result_t operator()(Args&&... args) const
    {
        RPY_THROW(std::runtime_error, "function not defined");
    }
};

template <typename T, typename=int>
struct AbsFunc : NoSuchFunction
{
};

template <typename T>
struct AbsFunc<T, void_t<decltype(abs(declval<T>()))>>
{
    using result_t = decltype(abs(declval<T>()));

    constexpr result_t operator()(const T& val) const noexcept(noexcept(abs(val)))
    {
        return abs(val);
    }
};

template <typename T, typename=int>
struct SqrtFunc : NoSuchFunction
{
};

template <typename T>
struct SqrtFunc<T, void_t<decltype(sqrt(declval<T>()))>>
{
    using result_t = decltype(sqrt(declval<T>()));

    constexpr result_t operator()(const T& val) const noexcept(noexcept(sqrt(val)))
    {
        return sqrt(val);
    }
};

template <typename T, typename=int>
struct PowFunc : NoSuchFunction
{
};

template <typename T>
struct PowFunc<T, void_t<decltype(sqrt(declval<T>()), std::declval<typename NumberTrait::exponent_t>())>>
{
    using exponent_t = typename NumberTrait::exponent_t;
    using result_t = decltype(pow(std::declval<T>()), std::declval<exponent_t>());

    constexpr result_t operator()(const T& val, exponent_t exponent) const noexcept(noexcept(pow(val, exponent)))
    {
        return pow(val, exponent);
    }
};

template <typename T, typename=int>
struct ExpFunc : NoSuchFunction
{
};

template <typename T>
struct ExpFunc<T, void_t<decltype(exp(declval<T>()))>>
{
    using result_t = decltype(exp(declval<T>()));

    constexpr result_t operator()(const T& val) const noexcept(noexcept(exp(val)))
    {
        return exp(val);
    }
};

template <typename T, typename=int>
struct LogFunc : NoSuchFunction
{
};

template <typename T>
struct LogFunc<T, void_t<decltype(log(declval<T>()))>>
{
    using result_t = decltype(log(declval<T>()));

    constexpr result_t operator()(const T& val) const noexcept(noexcept(log(val)))
    {
        return log(val);
    }
};



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
    *static_cast<typename Fn::result_t*>(dst) = fn(*static_cast<const T*>(src));
}
template <typename T>
void NumberTraitImpl<T>::unsafe_sqrt(void* dst, const void* src) const
{
    using Fn = number_trait_impl::SqrtFunc<T>;
    Fn fn;
    *static_cast<typename Fn::result_t*>(dst) = fn(*static_cast<const T*>(src));
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
    *static_cast<typename Fn::result_t*>(dst) = fn(*static_cast<const T*>(base));
}
template <typename T>
void NumberTraitImpl<T>::unsafe_exp(void* dst, const void* src) const
{
    using Fn = number_trait_impl::ExpFunc<T>;
    Fn fn;
    *static_cast<typename Fn::result_t*>(dst) = fn(*static_cast<const T*>(src));
}
template <typename T>
void NumberTraitImpl<T>::unsafe_log(void* dst, const void* src) const
{
    using Fn = number_trait_impl::LogFunc<T>;
    Fn fn;
    *static_cast<typename Fn::result_t*>(dst) = fn(*static_cast<const T*>(src));
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
