//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_COMPARISON_TRAIT_H
#define ROUGHPY_GENERICS_COMPARISON_TRAIT_H

#include "type_ptr.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "builtin_trait.h"
#include "roughpy/core/debug_assertion.h"
#include "type.h"

namespace rpy::generics {

class ConstRef;

/**
 * @class ComparisonTrait
 * @brief An abstract class defining traits for comparison operations.
 *
 * ComparisonTrait is a subclass of BuiltinTrait used to provide
 * a consistent interface for various comparison operations. It
 * ensures that derived classes provide implementations for specific
 * comparison operations between objects.
 */
class ROUGHPY_PLATFORM_EXPORT ComparisonTrait : public BuiltinTrait
{
    RPY_MAYBE_UNUSED const Type* p_type;

protected:
    constexpr explicit ComparisonTrait(const Type* type)
        : BuiltinTrait(my_id),
          p_type(type)
    {}

public:
    static constexpr BuiltinTraitID my_id = BuiltinTraitID::Comparison;

    virtual ~ComparisonTrait();

    /**
     * @brief Checks if a specific type of comparison operation is supported.
     *
     * This pure virtual function determines if the specific comparison
     * operation denoted by the given ComparisonType enumeration is supported by
     * the derived class.
     *
     * @param comp The type of comparison operation to check.
     * @return true if the specified comparison operation is supported,
     * otherwise false.
     */
    RPY_NO_DISCARD virtual bool has_comparison(ComparisonType comp
    ) const noexcept
            = 0;

    /**
     * @brief Compares two objects for equality in an unsafe manner.
     *
     * This pure virtual function performs an equality comparison between
     * objects pointed to by `lhs` and `rhs`. It is called "unsafe" because
     * it works with `void` pointers, which means that it does not perform
     * any type-checking at compile-time.
     *
     * Derived classes must provide the implementation for this function to
     * define how two objects of a specific type should be compared for
     * equality.
     *
     * @param lhs Pointer to the first object for comparison.
     * @param rhs Pointer to the second object for comparison.
     * @return true if the objects are considered equal according to the
     *         derived class implementation, otherwise false.
     */
    RPY_NO_DISCARD virtual bool
    unsafe_compare_equal(const void* lhs, const void* rhs) const noexcept
            = 0;
    /**
     * @brief Compares two objects to determine if the first is less than the
     * second, in an unsafe manner.
     *
     * This pure virtual function performs a "less-than" comparison between
     * objects pointed to by `lhs` and `rhs`. It is called "unsafe" because
     * it works with `void` pointers, which means that it does not perform
     * any type-checking at compile-time.
     *
     * Derived classes must provide the implementation for this function to
     * define how two objects of a specific type should be compared for order.
     *
     * @param lhs Pointer to the first object for comparison.
     * @param rhs Pointer to the second object for comparison.
     * @return true if the first object is considered less than the second
     *         according to the derived class implementation, otherwise false.
     */
    RPY_NO_DISCARD virtual bool
    unsafe_compare_less(const void* lhs, const void* rhs) const noexcept
            = 0;
    /**
     * @brief Compares two objects to determine if the first is less than or
     * equal to the second, in an unsafe manner.
     *
     * This pure virtual function performs a "less-than-or-equal-to" comparison
     * between objects pointed to by `lhs` and `rhs`. It is called "unsafe"
     * because it works with `void` pointers, which means that it does not
     * perform any type-checking at compile-time.
     *
     * Derived classes must provide the implementation for this function to
     * define how two objects of a specific type should be compared for order.
     *
     * @param lhs Pointer to the first object for comparison.
     * @param rhs Pointer to the second object for comparison.
     * @return true if the first object is considered less than or equal to the
     *         second according to the derived class implementation, otherwise
     * false.
     */
    RPY_NO_DISCARD virtual bool
    unsafe_compare_less_equal(const void* lhs, const void* rhs) const noexcept
            = 0;
    /**
     * @brief Compares two objects to determine if the first is greater than the
     * second, in an unsafe manner.
     *
     * This pure virtual function performs a "greater-than" comparison between
     * objects pointed to by `lhs` and `rhs`. It is called "unsafe"
     * because it works with `void` pointers, which means that it does not
     * perform any type-checking at compile-time.
     *
     * Derived classes must provide the implementation for this function to
     * define how two objects of a specific type should be compared for order.
     *
     * @param lhs Pointer to the first object for comparison.
     * @param rhs Pointer to the second object for comparison.
     * @return true if the first object is considered greater than the second
     *         according to the derived class implementation, otherwise false.
     */
    RPY_NO_DISCARD virtual bool
    unsafe_compare_greater(const void* lhs, const void* rhs) const noexcept
            = 0;
    /**
     * @brief Compares two objects to determine if the first is greater than or
     * equal to the second.
     *
     * This virtual function is intended to be overridden by derived classes to
     * provide specific comparison logic for determining if the first object is
     * greater than or equal to the second object.
     *
     * @param lhs Pointer to the first object to be compared.
     * @param rhs Pointer to the second object to be compared.
     * @return True if the first object is greater than or equal to the second
     * object, otherwise false.
     */
    RPY_NO_DISCARD virtual bool
    unsafe_compare_greater_equal(const void* lhs, const void* rhs)
            const noexcept
            = 0;

    /**
     * @brief Compares two objects for equality.
     *
     * This method checks if the two provided objects, represented by `lhs`
     * and `rhs`, are equal. It assumes that the objects are of the same type
     * and are both valid or invalid. For valid objects, it performs an
     * `unsafe_compare_equal` with their respective data.
     *
     * @param lhs A constant reference to the left-hand side object to compare.
     * @param rhs A constant reference to the right-hand side object to compare.
     * @return True if the objects are equal, false otherwise.
     */
    RPY_NO_DISCARD bool compare_equal(ConstRef lhs, ConstRef rhs) const;

    /**
     * @brief Compares two objects to determine if the first is less than the
     * second.
     *
     * This method checks if the object referenced by lhs is less than the
     * object referenced by rhs, based on the comparison traits implemented.
     *
     * @param lhs A constant reference to the first object in the comparison.
     * @param rhs A constant reference to the second object in the comparison.
     * @return True if lhs is less than rhs, otherwise false.
     */
    RPY_NO_DISCARD bool compare_less(ConstRef lhs, ConstRef rhs) const;

    /**
     * @brief Compares if one object is less than or equal to another.
     *
     * This method performs a comparison between two objects to determine if
     * the first object is less than or equal to the second object.
     *
     * @param lhs The left-hand side object reference to compare.
     * @param rhs The right-hand side object reference to compare.
     * @return True if the left-hand side object is less than or equal to the
     * right-hand side object; otherwise, false.
     */
    RPY_NO_DISCARD bool compare_less_equal(ConstRef lhs, ConstRef rhs) const;

    /**
     * @brief Compares two objects to determine if one is greater than the
     * other.
     *
     * This function checks if the necessary comparison operation is available
     * and that both objects are of the same type and not zero, then uses an
     * unsafe comparison to determine if the first object is greater than the
     * second.
     *
     * @param lhs The left-hand side object to compare.
     * @param rhs The right-hand side object to compare.
     * @return True if the left-hand side object is greater than the right-hand
     * side object, otherwise false.
     */
    RPY_NO_DISCARD bool compare_less_greater(ConstRef lhs, ConstRef rhs) const;

    /**
     * @brief Compares two objects for a less-than-or-equal, greater-than, or
     * equal relationship.
     *
     * This method checks if the object referred to by lhs is less than, greater
     * than, or equal to the object referred to by rhs. The comparison is
     * performed based on specific traits and ensures both objects are of the
     * same type and not zero.
     *
     * @param lhs The left-hand side object reference to be compared.
     * @param rhs The right-hand side object reference to be compared.
     * @return A boolean value indicating the result of the comparison.
     */
    RPY_NO_DISCARD bool
    compare_less_greater_equal(ConstRef lhs, ConstRef rhs) const;
};

template <typename T>
/**
 * @brief Concrete implementation of the ComparisonTrait interface.
 *
 * ComparisonTraitImpl provides specific implementations for comparison
 * operations between objects of a specified type. It overrides methods
 * to check the existence of comparison types and perform various unsafe
 * comparison operations such as equality, less than, less than or equal to,
 * greater than, and greater than or equal to.
 */
class ComparisonTraitImpl : public ComparisonTrait
{
public:
    explicit ComparisonTraitImpl(const Type* type);

    RPY_NO_DISCARD bool has_comparison(ComparisonType comp
    ) const noexcept override;

    RPY_NO_DISCARD bool unsafe_compare_equal(const void* lhs, const void* rhs)
            const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_less(const void* lhs, const void* rhs)
            const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_less_equal(
            const void* lhs,
            const void* rhs
    ) const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_greater(const void* lhs, const void* rhs)
            const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_greater_equal(
            const void* lhs,
            const void* rhs
    ) const noexcept override;
};

namespace dtl {

template <typename T>
using equals_test_t
        = decltype(std::declval<const T&>() == std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_equal_test_v = false;

template <typename T>
inline constexpr bool has_equal_test_v<T, void_t<equals_test_t<T>>> = true;

template <typename T>
using less_test_t
        = decltype(std::declval<const T&>() < std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_less_test_v = false;

template <typename T>
inline constexpr bool has_less_test_v<T, void_t<less_test_t<T>>> = true;

template <typename T>
using less_equal_test_t
        = decltype(std::declval<const T&>() <= std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_less_equal_test_v = false;

template <typename T>
inline constexpr bool has_less_equal_test_v<T, void_t<less_equal_test_t<T>>>
        = true;

template <typename T>
using greater_test_t
        = decltype(std::declval<const T&>() > std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_greater_test_v = false;

template <typename T>
inline constexpr bool has_greater_test_v<T, void_t<greater_test_t<T>>> = true;

template <typename T>
using greater_equal_test_t
        = decltype(std::declval<const T&>() >= std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_greater_equal_test_v = false;

template <typename T>
inline constexpr bool
        has_greater_equal_test_v<T, void_t<greater_equal_test_t<T>>>
        = true;

}// namespace dtl

template <typename T>
ComparisonTraitImpl<T>::ComparisonTraitImpl(const Type* type)
    : ComparisonTrait(type)
{
    RPY_DBG_ASSERT_EQ(type->type_info(), typeid(T));
}

template <typename T>
bool ComparisonTraitImpl<T>::has_comparison(ComparisonType comp) const noexcept
{
    switch (comp) {
        case ComparisonType::Equal: return dtl::has_equal_test_v<T>;
        case ComparisonType::Less: return dtl::has_less_test_v<T>;
        case ComparisonType::LessEqual:
            return dtl::has_less_equal_test_v<T>
                    || (dtl::has_less_test_v<T> && dtl::has_equal_test_v<T>);
        case ComparisonType::Greater: return dtl::has_greater_test_v<T>;
        case ComparisonType::GreaterEqual:
            return dtl::has_greater_equal_test_v<T>
                    || (dtl::has_greater_test_v<T> && dtl::has_equal_test_v<T>);
    }
    RPY_UNREACHABLE_RETURN(false);
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::Equal));
    return *static_cast<const T*>(lhs) == *static_cast<const T*>(rhs);
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_less(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::Less));
    return *static_cast<const T*>(lhs) < *static_cast<const T*>(rhs);
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_less_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::LessEqual));
    if constexpr (dtl::has_less_equal_test_v<T>) {
        return *static_cast<const T*>(lhs) <= *static_cast<const T*>(rhs);
    } else {
        return (*static_cast<const T*>(lhs) < *static_cast<const T*>(rhs))
                || (*static_cast<const T*>(lhs) == *static_cast<const T*>(rhs));
    }
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_greater(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::Greater));
    return *static_cast<const T*>(lhs) > *static_cast<const T*>(rhs);
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_greater_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::GreaterEqual));
    if constexpr (dtl::has_greater_equal_test_v<T>) {
        return *static_cast<const T*>(lhs) >= *static_cast<const T*>(rhs);
    } else {
        return (*static_cast<const T*>(lhs) > *static_cast<const T*>(rhs))
                || (*static_cast<const T*>(lhs) == *static_cast<const T*>(rhs));
    }
}

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_COMPARISON_TRAIT_H
