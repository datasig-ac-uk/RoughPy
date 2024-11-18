//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_BUILTIN_TRAIT_H
#define ROUGHPY_GENERICS_BUILTIN_TRAIT_H

#include "roughpy/core/check.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {


class ComparisonTrait;
class ArithmeticTrait;
class NumberTrait;

/**
 * Enum representing the IDs of various built-in traits.
 *
 * Each ID corresponds to a specific trait that can be used by the system.
 */
enum class BuiltinTraitID : size_t
{
    Comparison = 0,
    Arithmetic,
    Number
};

/**
 * Enum representing arithmetic operations.
 *
 * This enum lists the basic arithmetic operations that can be performed.
 */
enum class ArithmeticOperation
{
    Add = 0,
    Sub,
    Mul,
    Div
};

/**
 * Enum representing the types of comparisons that can be performed.
 *
 * This enum lists various comparison operations such as equality and
 * inequality.
 */
enum class ComparisonType
{
    Equal = 0,
    Less,
    LessEqual,
    Greater,
    GreaterEqual
};

/**
 * Enum representing functions that can be performed on numbers.
 *
 * This enum lists various mathematical functions such as absolute value,
 * square root, power, exponential, logarithm, and functions to retrieve
 * the real and imaginary parts of a number.
 */
enum class NumberFunction
{
    Abs = 0,
    Sqrt,
    Pow,
    Exp,
    Log,
    FromRational,
    Real,
    Imaginary  // Considered together with Real
};

using exponent_t = int;

/**
 * Class representing a built-in trait.
 *
 * This class encapsulates a built-in trait, identified by its ID. It provides
 * an interface to retrieve the ID of the trait.
 */
class ROUGHPY_PLATFORM_EXPORT BuiltinTrait {
    BuiltinTraitID m_id;

protected:

    explicit constexpr BuiltinTrait(BuiltinTraitID id) : m_id(id) {}
public:

    RPY_NO_DISCARD
    constexpr BuiltinTraitID id() const noexcept { return m_id; }

};


template <typename TraitObject>
/**
 * Casts a built-in trait to a specific trait object type.
 *
 * This function attempts to cast a given pointer to a BuiltinTrait to a
 * pointer to a specific trait object type. If the trait is null or its ID
 * does not match the ID of the specified trait object type, the function
 * returns null.
 *
 * @tparam TraitObject The specific trait object type to cast to.
 * @param trait The pointer to the BuiltinTrait that is to be casted.
 * @return A pointer to the specific trait object type, or null if the
 *         cast is not possible.
 */
constexpr enable_if_t<is_base_of_v<BuiltinTrait, TraitObject>, const TraitObject*>
trait_cast(const BuiltinTrait* trait) noexcept
{
    if (trait == nullptr || trait->id() != TraitObject::my_id) {
        return nullptr;
    }
    return static_cast<const TraitObject*>(trait);
}


}

#endif //ROUGHPY_GENERICS_BUILTIN_TRAIT_H
