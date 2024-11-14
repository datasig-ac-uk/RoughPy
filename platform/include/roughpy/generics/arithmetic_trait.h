//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H
#define ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H

#include "roughpy/generics/builtin_trait.h"
#include "type_ptr.h"

namespace rpy::generics {

class ConstRef;
class Ref;
class Value;

class ROUGHPY_PLATFORM_EXPORT ArithmeticTrait : public BuiltinTrait {
    TypePtr p_type;
    TypePtr p_rational_type;
protected:

    explicit constexpr ArithmeticTrait(TypePtr type, TypePtr rational_type)
        : BuiltinTrait(my_id),
          p_type(std::move(type)),
          p_rational_type(std::move(rational_type))
    {}

public:

    static constexpr BuiltinTraitID my_id = BuiltinTraitID::Arithmetic;

    virtual ~ArithmeticTrait();

    enum class Operation
    {
        Add,
        Sub,
        Mul,
        Div
    };

    RPY_NO_DISCARD
    virtual bool has_operation(Operation op) const noexcept;

    virtual void unsafe_add_inplace(void* lhs, const void* rhs) const = 0;
    virtual void unsafe_sub_inplace(void* lhs, const void* rhs) const = 0;
    virtual void unsafe_mul_inplace(void* lhs, const void* rhs) const = 0;
    virtual void unsafe_div_inplace(void* lhs, const void* rhs) const = 0;

    RPY_NO_DISCARD
    const TypePtr& type() const noexcept { return p_type; }

    RPY_NO_DISCARD
    const TypePtr& rational_type() const noexcept { return p_rational_type; }

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


template <typename T, typename R>
class ROUGHPY_PLATFORM_NO_EXPORT ArithmeticTraitImpl
    : public ArithmeticTrait
{
public:



};


}

#endif //ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H
