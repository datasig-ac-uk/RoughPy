//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_COMPARISON_TRAIT_H
#define ROUGHPY_GENERICS_COMPARISON_TRAIT_H

#include "type_ptr.h"


#include "roughpy/platform/roughpy_platform_export.h"

#include "builtin_trait.h"

namespace rpy::generics {

class ConstRef;

class ROUGHPY_PLATFORM_EXPORT ComparisonTrait : public BuiltinTrait
{
protected:

    constexpr ComparisonTrait() : BuiltinTrait(my_id)
    {}

public:

    static constexpr BuiltinTraitID my_id = BuiltinTraitID::Comparison;

    enum class Comparison
    {
        Equal,
        Less,
        LessEqual,
        Greater,
        GreaterEqual
    };

    virtual ~ComparisonTrait();

    RPY_NO_DISCARD
    virtual bool has_comparison(Comparison comp) const noexcept = 0;

    RPY_NO_DISCARD
    virtual bool unsafe_compare_equal(const void* lhs, const void* rhs) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool unsafe_compare_less(const void* lhs, const void* rhs) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool unsafe_compare_less_equal(const void* lhs, const void* rhs) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool unsafe_compare_greater(const void* lhs, const void* rhs) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool unsafe_compare_greater_equal(const void* lhs, const void* rhs) const noexcept = 0;

    RPY_NO_DISCARD
    bool compare_equal(ConstRef lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    bool compare_less(ConstRef lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    bool compare_less_equal(ConstRef lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    bool compare_less_greater(ConstRef lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    bool compare_less_greater_equal(ConstRef lhs, ConstRef rhs) const;

};




}


#endif //ROUGHPY_GENERICS_COMPARISON_TRAIT_H
