//
// Created by sam on 09/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_H
#define ROUGHPY_GENERICS_TRAITS_H

#include "roughpy/core/hash.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/errors.h"


namespace rpy::generics {


class ConstReference;
class Reference;
class Value;
class Type;

using TypePtr = const Type*;


enum class TraitType {
    Builtin,
    Static,
    Dynamic
};

class Trait {
public:

    static constexpr TraitType Builtin = TraitType::Builtin;
    static constexpr TraitType Static = TraitType::Static;
    static constexpr TraitType Dynamic = TraitType::Dynamic;

    virtual ~Trait() = default;

    RPY_NO_DISCARD
    virtual TraitType type() const noexcept = 0;

    RPY_NO_DISCARD
    virtual string_view name() const noexcept = 0;
};


class BuiltinTrait : public Trait {
public:

    static constexpr size_t hashable = 0;
    static constexpr size_t equality = 1;
    static constexpr size_t arithmetic = 2;
    static constexpr size_t ordering = 3;
    static constexpr size_t number = 4;

    static constexpr TraitType this_type = TraitType::Builtin;

    RPY_NO_DISCARD
    TraitType type() const noexcept final
    {
        return this_type;
    }

    RPY_NO_DISCARD
    virtual size_t index() const noexcept = 0;

};


class StaticTrait : public Trait
{
public:
    static constexpr TraitType this_type = TraitType::Static;

    RPY_NO_DISCARD
    TraitType type() const noexcept final { return this_type; }

    RPY_NO_DISCARD
    virtual string_view id() const noexcept = 0;

};



class DynamicTrait : public Trait
{

public:
    static constexpr TraitType this_type = TraitType::Dynamic;

    RPY_NO_DISCARD TraitType type() const noexcept final
    {
        return this_type;
    }

    RPY_NO_DISCARD
    virtual string_view id() const noexcept = 0;
};



class Hashable : public BuiltinTrait
{
public:

    static constexpr string_view this_name = "Hashable";

    RPY_NO_DISCARD
    string_view name() const noexcept final
    {
        return this_name;
    }

    RPY_NO_DISCARD size_t index() const noexcept final
    {
        return hashable;
    }

    RPY_NO_DISCARD virtual hash_t hash(ConstReference value) const noexcept = 0;
};



class Equality : public BuiltinTrait
{
public:
    static constexpr string_view this_name = "Equality";

    RPY_NO_DISCARD
    string_view name() const noexcept final
    {
        return this_name;
    }

    RPY_NO_DISCARD size_t index() const noexcept final
    {
        return equality;
    }

    RPY_NO_DISCARD virtual bool is_equal(ConstReference lhs, ConstReference rhs) const noexcept = 0;
    RPY_NO_DISCARD bool not_equal(ConstReference lhs, ConstReference rhs) const noexcept;
};



class Arithmetic : public BuiltinTrait
{
public:

    static constexpr string_view this_name = "Arithmetic";

    RPY_NO_DISCARD
    string_view name() const noexcept final { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final
    {
        return arithmetic;
    }

    virtual void add_inplace(Reference lhs, ConstReference rhs) const = 0;
    virtual void sub_inplace(Reference lhs, ConstReference rhs) const = 0;
    virtual void mul_inplace(Reference lhs, ConstReference rhs) const = 0;
    virtual void div_inplace(Reference lhs, ConstReference rhs) const = 0;

    RPY_NO_DISCARD
    virtual Value add(ConstReference lhs, ConstReference rhs) const;
    RPY_NO_DISCARD
    virtual Value sub(ConstReference lhs, ConstReference rhs) const;
    RPY_NO_DISCARD
    virtual Value mul(ConstReference lhs, ConstReference rhs) const;
    RPY_NO_DISCARD
    virtual Value div(ConstReference lhs, ConstReference rhs) const;
};


class Ordering : public BuiltinTrait
{
public:

    enum class OrderingResult
    {
        Incomparable,
        Equal,
        LessThan,
        GreaterThan,
    };

    static constexpr string_view this_name = "Ordering";


    RPY_NO_DISCARD
    string_view name() const noexcept final { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final
    {
        return ordering;
    }

    RPY_NO_DISCARD
    virtual OrderingResult compare(ConstReference lhs, ConstReference rhs) const noexcept;

    RPY_NO_DISCARD
    bool equals(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD
    bool less(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD
    bool less_equal(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD
    bool greater(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD
    bool greater_equal(ConstReference lhs, ConstReference rhs) const noexcept;
    RPY_NO_DISCARD
    bool comparable(ConstReference lhs, ConstReference rhs) const noexcept;
};


class Number : public BuiltinTrait
{

public:

    using power_t = int;

    static constexpr string_view this_name = "Number";

    RPY_NO_DISCARD
    string_view name() const noexcept final { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final
    {
        return number;
    }

    RPY_NO_DISCARD
    virtual TypePtr rational_type(const Type& type) const noexcept = 0;

    RPY_NO_DISCARD
    virtual TypePtr real_type(const Type& type) const noexcept
    {
        return &type;
    }

    RPY_NO_DISCARD
    virtual TypePtr imaginary_type(const Type& type) const noexcept
    {
        return nullptr;
    }

    RPY_NO_DISCARD
    virtual Value real(ConstReference value) const;

    RPY_NO_DISCARD
    virtual Value imaginary(ConstReference value) const;

    RPY_NO_DISCARD
    virtual Value minus(ConstReference value) const;

    RPY_NO_DISCARD
    virtual Value abs(ConstReference value) const;

    RPY_NO_DISCARD
    virtual Value sqrt(ConstReference value) const;

    RPY_NO_DISCARD
    virtual Value pow(ConstReference value, power_t power) const;

    RPY_NO_DISCARD
    virtual Value exp(ConstReference value) const;

    RPY_NO_DISCARD
    virtual Value log(ConstReference value) const;

    RPY_NO_DISCARD
    virtual Value from_rational(int64_t numerator, int64_t denominator) const;
};



















template <typename T>
inline constexpr bool is_trait_base_v = is_same_v<T, BuiltinTrait>
    || is_same_v<T, StaticTrait>
    || is_same_v<T, DynamicTrait>;

template <typename T>
enable_if_t<is_trait_base_v<T> && !is_const_v<T>, add_lvalue_reference_t<T>>
trait_cast(Trait& trait)
{
    if (trait.type() != T::this_type) {
        RPY_THROW(std::bad_cast);
    }
    return static_cast<T&>(trait);
}

template <typename T>
enable_if_t<is_trait_base_v<T>, add_lvalue_reference_t<add_const_t<T>>>
trait_cast(const Trait& trait)
{
    if (trait.type() != T::this_type) {
        RPY_THROW(std::bad_cast);
    }
    return static_cast<const T&>(trait);
}







}

#endif //ROUGHPY_GENERICS_TRAITS_H
