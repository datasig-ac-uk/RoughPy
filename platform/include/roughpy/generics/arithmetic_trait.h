//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H
#define ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H

#include "roughpy/core/debug_assertion.h"

#include "builtin_trait.h"
#include "type_ptr.h"

namespace rpy::generics {

class ConstRef;
class Ref;
class Value;

class ROUGHPY_PLATFORM_EXPORT ArithmeticTrait : public BuiltinTrait {
    // Builtin traits should be held on the Type class itself, so having
    // TypePtrs here would cause a reference loop.
    const Type* p_type;
    const Type* p_rational_type;
protected:

    explicit ArithmeticTrait(const Type* type, const Type* rational_type)
        : BuiltinTrait(my_id),
          p_type(type),
          p_rational_type(rational_type)
    {}

public:

    static constexpr auto my_id = BuiltinTraitID::Arithmetic;

    virtual ~ArithmeticTrait();



    RPY_NO_DISCARD
    virtual bool has_operation(ArithmeticOperation op) const noexcept;

    virtual void unsafe_add_inplace(void* lhs, const void* rhs) const noexcept = 0;
    virtual void unsafe_sub_inplace(void* lhs, const void* rhs) const = 0;
    virtual void unsafe_mul_inplace(void* lhs, const void* rhs) const = 0;
    virtual void unsafe_div_inplace(void* lhs, const void* rhs) const = 0;

    RPY_NO_DISCARD
    const Type* type() const noexcept { return p_type; }

    RPY_NO_DISCARD
    const Type* rational_type() const noexcept { return p_rational_type; }

    void add_inplace(Ref lhs, ConstRef rhs) const;
    void sub_inplace(Ref lhs, ConstRef rhs) const;
    void mul_inplace(Ref lhs, ConstRef rhs) const;
    void div_inplace(Ref lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    Value add(ConstRef lhs, ConstRef rhs) const;
    RPY_NO_DISCARD
    Value sub(ConstRef lhs, ConstRef rhs) const;
    RPY_NO_DISCARD
    Value mul(ConstRef lhs, ConstRef rhs) const;
    RPY_NO_DISCARD
    Value div(ConstRef lhs, ConstRef rhs) const;

};


template <typename T, typename R=T>
class ArithmeticTraitImpl : public ArithmeticTrait
{
public:

    ArithmeticTraitImpl(const Type* type, const Type* rational_type);

    RPY_NO_DISCARD bool has_operation(ArithmeticOperation op) const noexcept override;

    void unsafe_add_inplace(void* lhs, const void* rhs) const noexcept override;
    void unsafe_sub_inplace(void* lhs, const void* rhs) const noexcept override;
    void unsafe_mul_inplace(void* lhs, const void* rhs) const noexcept override;
    void unsafe_div_inplace(void* lhs, const void* rhs) const noexcept override;
};


namespace dtl {

template <typename U>
using add_result_t = decltype(std::declval<U&>() += std::declval<const U&>());

template <typename U>
using sub_result_t = decltype(std::declval<U&>() -= std::declval<const U&>());

template <typename U>
using mul_result_t = decltype(std::declval<U&>() *= std::declval<const U&>());

template <typename U, typename V>
using div_result_t = decltype(std::declval<U&>() /= std::declval<const V&>());

template <typename U, typename = void>
inline constexpr bool has_add_v = false;

template <typename U>
inline constexpr bool has_add_v<U, void_t<add_result_t<U>>> = true;

template <typename U, typename = void>
inline constexpr bool has_sub_v = false;

template <typename U>
inline constexpr bool has_sub_v<U, void_t<sub_result_t<U>>> = true;

template <typename U, typename = void>
inline constexpr bool has_mul_v = false;

template <typename U>
inline constexpr bool has_mul_v<U, void_t<mul_result_t<U>>> = true;

template <typename U, typename V, typename = void>
inline constexpr bool has_div_v = false;

template <typename U, typename V>
inline constexpr bool has_div_v<U, V, void_t<div_result_t<U, V>>> = true;

}// namespace dtl


template <typename T, typename R>
ArithmeticTraitImpl<T, R>::ArithmeticTraitImpl(
        const Type* type,
        const Type* rational_type
)
    : ArithmeticTrait(type, rational_type)
{}

template <typename T, typename R>
bool ArithmeticTraitImpl<T, R>::has_operation(ArithmeticOperation op) const noexcept
{
    switch (op) {
        case ArithmeticOperation::Add:
            return dtl::has_add_v<T>;
        case ArithmeticOperation::Sub:
            return dtl::has_sub_v<T>;
        case ArithmeticOperation::Mul:
            return dtl::has_mul_v<T>;
        case ArithmeticOperation::Div:
            return dtl::has_div_v<T, R>;
    }
    RPY_UNREACHABLE_RETURN(false);
}

template <typename T, typename R>
void ArithmeticTraitImpl<T, R>::unsafe_add_inplace(void* lhs, const void*
rhs) const noexcept
{
    RPY_DBG_ASSERT(dtl::has_add_v<T>);
    *static_cast<T*>(lhs) += *static_cast<const T*>(rhs);
}

template <typename T, typename R>
void ArithmeticTraitImpl<T, R>::unsafe_sub_inplace(void* lhs, const void* rhs)
        const noexcept
{
    RPY_DBG_ASSERT(dtl::has_add_v<T>);
    *static_cast<T*>(lhs) -= *static_cast<const T*>(rhs);
}

template <typename T, typename R>
void ArithmeticTraitImpl<T, R>::unsafe_mul_inplace(void* lhs, const void* rhs)
        const noexcept
{
    RPY_DBG_ASSERT(dtl::has_add_v<T>);
    *static_cast<T*>(lhs) *= *static_cast<const T*>(rhs);
}

template <typename T, typename R>
void ArithmeticTraitImpl<T, R>::unsafe_div_inplace(void* lhs, const void* rhs)
        const noexcept
{
    RPY_DBG_ASSERT(dtl::has_add_v<T>);
    *static_cast<T*>(lhs) /= *static_cast<const R*>(rhs);
}




}// namespace rpy::generics

#endif //ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H
