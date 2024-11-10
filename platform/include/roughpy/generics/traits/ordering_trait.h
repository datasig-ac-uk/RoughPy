//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_ORDERING_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_ORDERING_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "builtin_trait.h"

namespace rpy::generics {

class ConstReference;



class Ordering : public BuiltinTrait
{
public:
    static constexpr string_view this_name = "Ordering";

    RPY_NO_DISCARD string_view name() const noexcept final { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final { return ordering; }

    enum class Result
    {
        Incomparable,
        Equal,
        LessThan,
        GreaterThan,
    };

    static constexpr Result ResultIncomparable = Result::Incomparable;
    static constexpr Result ResultEqual = Result::Equal;
    static constexpr Result ResultLessThan = Result::LessThan;
    static constexpr Result ResultGreaterThan = Result::GreaterThan;

    RPY_NO_DISCARD virtual Result
    compare(ConstReference lhs, ConstReference rhs) const noexcept;

    RPY_NO_DISCARD bool
    equals(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD bool
    less(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD bool
    less_equal(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD bool
    greater(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD bool
    greater_equal(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD bool
    comparable(ConstReference lhs, ConstReference rhs) const noexcept;
};

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_TRAITS_ORDERING_TRAIT_H
