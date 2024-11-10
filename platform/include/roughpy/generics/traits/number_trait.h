//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_NUMBER_TRAIT_H
#define ROUGHPY_GENERICS_NUMBER_NUMBER_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/generics/type_ptr.h"

#include "builtin_trait.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {

class ConstReference;
class Value;


class ROUGHPY_PLATFORM_EXPORT Number : public BuiltinTrait
{

public:

    static constexpr string_view this_name = "Number";

    RPY_NO_DISCARD string_view name() const noexcept final { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final { return number; }

    RPY_NO_DISCARD virtual TypePtr rational_type(const Type& type
    ) const noexcept
            = 0;

    RPY_NO_DISCARD virtual TypePtr real_type(const Type& type) const noexcept
    {
        return &type;
    }

    RPY_NO_DISCARD virtual TypePtr imaginary_type(const Type& type
    ) const noexcept
    {
        return nullptr;
    }

    RPY_NO_DISCARD virtual Value real(ConstReference value) const;

    RPY_NO_DISCARD virtual Value imaginary(ConstReference value) const;

    RPY_NO_DISCARD virtual Value minus(ConstReference value) const;

    RPY_NO_DISCARD virtual Value abs(ConstReference value) const;

    RPY_NO_DISCARD virtual Value sqrt(ConstReference value) const;

    RPY_NO_DISCARD virtual Value pow(ConstReference value, ConstReference power) const;

    RPY_NO_DISCARD virtual Value exp(ConstReference value) const;

    RPY_NO_DISCARD virtual Value log(ConstReference value) const;

    RPY_NO_DISCARD virtual Value
    from_rational(int64_t numerator, int64_t denominator) const;
};

}


#endif //ROUGHPY_GENERICS_TRAITS_NUMBER_TRAIT_H
