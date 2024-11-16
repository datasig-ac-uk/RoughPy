//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_NUMBER_TRAIT_H
#define ROUGHPY_GENERICS_NUMBER_TRAIT_H


#include <roughpy/core/debug_assertion.h>

#include "builtin_trait.h"
#include "type_ptr.h"

namespace rpy::generics {

class ConstRef;
class Ref;
class Value;

class ROUGHPY_PLATFORM_EXPORT NumberTrait : public BuiltinTrait
{
    TypePtr p_type;
    TypePtr p_real_type;

protected:

    NumberTrait(TypePtr type, TypePtr real_type=nullptr)
        : BuiltinTrait(my_id),
          p_type(std::move(type)),
          p_real_type(std::move(real_type))
    {}

public:

    static constexpr auto my_id = BuiltinTraitID::Number;



    virtual ~NumberTrait();

    virtual void unsafe_real(void* dst, const void* src) const;
    virtual void unsafe_imaginary(void* dst, const void* src) const;

    virtual void unsafe_abs(void* dst, const void* src) const noexcept = 0;


    virtual void unsafe_sqrt(void* dst, const void* src) const;
    virtual void unsafe_pow(void* dst, const void* base, int64_t exponent) const;

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
    RPY_NO_DISCARD Value pow(ConstRef value, int64_t power) const;
    RPY_NO_DISCARD Value exp(ConstRef value) const;
    RPY_NO_DISCARD Value log(ConstRef value) const;

    RPY_NO_DISCARD Value
    from_rational(int64_t numerator, int64_t denominator) const;

    RPY_NO_DISCARD
    const TypePtr& real_type() const noexcept
    {
        return p_real_type ? p_real_type : p_type;
    }

    RPY_NO_DISCARD const TypePtr& imaginary_type() const noexcept
    {
        return p_real_type;
    }
};

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_NUMBER_TRAIT_H
