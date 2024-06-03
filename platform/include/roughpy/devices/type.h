//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_DEVICES_TYPE_H
#define ROUGHPY_DEVICES_TYPE_H

#include "core.h"

#include <roughpy/core/container/unordered_map.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/sync.h>
#include <roughpy/core/types.h>

#ifndef RPY_NO_RTTI

#  include <typeinfo>

#endif

namespace rpy {
namespace devices {

/**
 * @brief Type traits for a type.
 *
 * The TypeTraits struct provides information about various traits of a type.
 *
 * @var standard_layout
 * Indicates whether the type has a standard layout. A type has a standard
 * layout if it satisfies all the requirements of standard layout types. A
 * standard layout type has no virtual functions, no non-static data members of
 * reference type, no base classes, and all base class subobjects have the same
 * access specifier.
 *
 * @var trivially_copyable
 * Indicates whether the type can be copied using a simple bitwise copy. A type
 * is trivially copyable if it has a trivial copy constructor and a trivial copy
 * assignment operator. A trivial copy constructor is a constructor that
 * performs a simple bitwise copy of the object representation. A trivial copy
 * assignment operator is an operator that performs a simple bitwise copy of the
 * object representation.
 *
 * @var trivially_constructible
 * Indicates whether the type can be constructed in a trivial manner. A type is
 * trivially constructible if it has a trivial default constructor, a trivial
 * copy constructor, and a trivial move constructor.
 *
 * @var trivially_default_constructible
 * Indicates whether the type can be default constructed in a trivial manner. A
 * type is trivially default constructible if it has a trivial default
 * constructor.
 *
 * @var trivially_copy_constructible
 * Indicates whether the type can be copy constructed in a trivial manner. A
 * type is trivially copy constructible if it has a trivial copy constructor.
 *
 * @var trivially_copy_assignable
 * Indicates whether the type can be copy assigned in a trivial manner. A type
 * is trivially copy assignable if it has a trivial copy assignment operator.
 *
 * @var trivially_destructible
 * Indicates whether the type can be destructed in a trivial manner. A type is
 * trivially destructible if it has a trivial destructor.
 *
 * @var polymorphic
 * Indicates whether the type is a polymorphic type. A polymorphic type is a
 * type that has at least one virtual function.
 *
 * @var is_signed
 * Indicates whether the type is a signed integral type.
 *
 * @var is_floating_point
 * Indicates whether the type is a floating-point type.
 *
 * @var is_integral
 * Indicates whether the type is an integral type.
 */
struct TypeTraits {
    bool standard_layout : 1;
    bool trivially_copyable : 1;
    bool trivially_constructible : 1;
    bool trivially_default_constructible : 1;
    bool trivially_copy_constructible : 1;
    bool trivially_copy_assignable : 1;
    bool trivially_destructible : 1;
    bool polymorphic : 1;
    bool is_signed : 1;
    bool is_floating_point : 1;
    bool is_integral : 1;
};

enum class TypeComparison
{
    AreSame,
    TriviallyConvertible,
    Convertible,
    NotConvertible
};

namespace type_support {
using MathFn = void (*)(void*, const void*);
using CompareFn = bool (*)(void*, const void*) noexcept;
using IsZeroFn = bool (*)(const void*) noexcept;
using PowFn = void (*)(void*, const void*, unsigned) noexcept;

struct TypeArithmetic {
    std::function<void(void*, const void*)> add_inplace;
    std::function<void(void*, const void*)> sub_inplace;
    std::function<void(void*, const void*)> mul_inplace;
    std::function<void(void*, const void*)> div_inplace;
};

struct TypeConversions {
    std::function<void(void*, const void*)> convert;
    std::function<void(void*, void*)> move_convert;
};

struct TypeComparisons {
    std::function<bool(const void*, const void*)> equals;
    std::function<bool(const void*, const void*)> less;
    std::function<bool(const void*, const void*)> less_equal;
    std::function<bool(const void*, const void*)> greater;
    std::function<bool(const void*, const void*)> greater_equal;
    std::function<bool(const void*)> is_zero;
};

struct NumTraits {
    TypePtr rational_type;
    TypePtr real_type;
    TypePtr imag_type;

    MathFn abs;
    MathFn sqrt;
    MathFn real;
    MathFn imag;
    MathFn conj;

    PowFn pow;

    MathFn exp;
    MathFn log;
};

}// namespace type_support

struct TypeSupport {
    type_support::TypeArithmetic arithmetic;
    type_support::TypeComparisons comparison;
    type_support::TypeConversions conversions;
};

/**
 * @class Type
 * @brief Represents a type with various traits and behavior.
 *
 * The Type class provides information about a type, including its name, unique
 * id, type information, and type traits. It also provides various methods for
 * working with the type, such as allocating memory for scalars, checking type
 * compatibility, and converting between types.
 */
class ROUGHPY_DEVICES_EXPORT Type
{
    string_view m_id;
    string_view m_name;
    TypeInfo m_info;
    TypeTraits m_traits;

    class TypeSupportDispatcher;

    std::unique_ptr<TypeSupportDispatcher> p_type_support;

    std::unique_ptr<type_support::NumTraits> p_num_traits;

protected:
    type_support::NumTraits& setup_num_traits()
    {
        p_num_traits = std::make_unique<type_support::NumTraits>();
        return *p_num_traits;
    }

public:
    /**
     * @brief Construct a new Type object
     *
     * @param id The unique internal ID string for this type
     * @param name The name of this type
     * @param info The type information
     * @param traits The type traits of this type
     */
    explicit
    Type(string_view id, string_view name, TypeInfo info, TypeTraits traits);

    Type() = delete;
    Type(const Type&) = delete;
    Type(Type&&) noexcept = delete;

    virtual Type& operator=(const Type&) = delete;
    virtual Type& operator=(Type&&) noexcept = delete;

    virtual ~Type();

    /**
     * @brief Get the name of this type
     */
    RPY_NO_DISCARD string_view name() const noexcept { return m_name; }

    /**
     * @brief Get the unique internal ID string for this type
     * @return const reference to the ID string.
     */
    RPY_NO_DISCARD string_view id() const noexcept { return m_id; }

    /**
     * @brief Retrieves the type information of the object.
     *
     * @return The type information of the object.
     */
    RPY_NO_DISCARD TypeInfo type_info() const noexcept { return m_info; }

    /**
     * @brief Get the type traits of this object
     *
     * This method returns the type traits of the object that implements this
     * method.
     *
     * @return The const reference to the TypeTraits object representing the
     * type traits of this object
     *
     * @note This method is declared as virtual to allow derived classes to
     * override the behavior. The noexcept specifier indicates that this method
     * will not throw any exception.
     */
    RPY_NO_DISCARD virtual const TypeTraits& type_traits() const noexcept
    {
        return m_traits;
    }

    /**
     * @brief Allocate new scalars in memory
     * @param device Device on which data should be allocated
     * @param count Number of scalars to allocate space
     * @return ScalarPointer pointing to the newly allocated raw memory.
     *
     * Note that ScalarArrays are internally reference counted, so will
     * remain valid whilst there is a ScalarArray object with that data.
     */
    RPY_NO_DISCARD virtual Buffer allocate(Device device, dimn_t count) const;

    /**
     * @brief Allocate single scalar pointer for a Scalar.
     *
     * Only necessary for large scalar types.
     */
    RPY_NO_DISCARD virtual void* allocate_single() const;

    /**
     * @brief Free a previously allocated single scalar value.
     *
     * Only necessary for large scalar types
     */
    virtual void free_single(void* ptr) const;

    /**
     * @brief Check if the current object supports the specified device.
     *
     * This method checks if the current object supports the specified device.
     *
     * @param device The device to check support for.
     *
     * @return true if the current object supports the specified device, false
     * otherwise.
     *
     * @note This method is declared as virtual to allow derived classes to
     * override the behavior. The noexcept specifier indicates that this method
     * will not throw any exception.
     */
    RPY_NO_DISCARD virtual bool supports_device(const Device& device
    ) const noexcept;

    /**
     * @brief Check if the current object is convertible to the specified
     * destination type.
     *
     * @param dest_type A pointer to the destination type to check if the
     * current object is convertible to.
     * @return True if the current object is convertible to the destination
     * type, false otherwise.
     */
    RPY_NO_DISCARD virtual bool convertible_to(const Type* dest_type
    ) const noexcept;

    /**
     * @brief Check if this type is convertible from another type.
     *
     * This method checks if the current type can be converted from the
     * specified source type.
     *
     * @param src_type The source type to check compatibility with.
     *
     * @return true if the current type is convertible from the source type,
     * false otherwise.
     */
    RPY_NO_DISCARD virtual bool convertible_from(const Type* src_type
    ) const noexcept;

    /**
     * @brief Compares the current Type object with another Type object.
     *
     * This method compares the current Type object with another Type object,
     * and returns the result of the comparison.
     *
     * @param other A pointer to the Type object to compare with.
     *
     * @return The result of the comparison:
     * - TypeComparison::AreSame if the two Type objects are the same.
     * - TypeComparison::TriviallyConvertible if the other Type object is an
     * arithmetic type.
     * - TypeComparison::Convertible if the other Type object is convertible to
     * the current Type object.
     * - TypeComparison::NotConvertible if the other Type object is not
     * convertible to the current Type object.
     *
     * @note This method is declared as constant, indicating that it does not
     * modify the state of the object. The noexcept specifier indicates that
     * this method will not throw any exceptions.
     */
    RPY_NO_DISCARD virtual TypeComparison compare_with(const Type* other
    ) const noexcept;

    /**
     * @brief Get the number of bytes.
     *
     * Returns the number of bytes of the object.
     *
     * @return The number of bytes.
     */
    RPY_NO_DISCARD dimn_t bytes() const noexcept { return m_info.bytes; }

    /**
     * @brief Returns the minimum alignment of the type.
     *
     * The `min_alignment` method returns the minimum alignment of the type. The
     * alignment specifies the memory alignment requirement for objects of the
     * type.
     *
     * @return The minimum alignment of the type.
     *
     * @see type.h
     */
    RPY_NO_DISCARD dimn_t min_alignment() const noexcept
    {
        return m_info.alignment;
    }

    /**
     * @brief Copies elements from source buffer to destination buffer.
     *
     * This method copies elements from the source buffer to the destination
     * buffer.
     *
     * @param dst Pointer to the destination buffer.
     * @param src Pointer to the source buffer.
     * @param count Number of elements to copy.
     *
     * @note The size of the source and destination buffers must be large enough
     * to accommodate the number of elements to copy.
     *
     * @note This method is virtual and can be overridden by derived classes.
     */
    virtual void copy(void* dst, const void* src, dimn_t count) const;

    /**
     * @brief Moves a block of memory from one location to another.
     *
     * The move function is used to move a block of memory from the source
     * address to the destination address. The size of the block of memory is
     * specified in the count parameter.
     *
     * @param dst The destination address where the block of memory will be
     * moved to.
     * @param src The source address where the block of memory will be moved
     * from.
     * @param count The size of the block of memory in bytes.
     */
    virtual void move(void* dst, void* src, dimn_t count) const;

    /**
     * @brief Returns the TypeArithmetic object for performing arithmetic
     * operations with the given Type.
     *
     * The `arithmetic` method returns the TypeArithmetic object that provides
     * support for performing arithmetic operations with the given Type and
     * another type specified by `other_type`.
     *
     * @param other_type A pointer to the Type object representing the other
     * type for arithmetic operations.
     *
     * @return The TypeArithmetic object that encapsulates the arithmetic
     * operations supported by the Type and the `other_type`.
     *
     * @remark The returned TypeArithmetic object can be used to perform
     * arithmetic operations such as addition, subtraction, multiplication, and
     * division between values of the Type and `other_type`.
     *
     * @see TypeArithmetic
     */
    RPY_NO_DISCARD const type_support::TypeArithmetic&
    arithmetic(const Type* other_type) const;

    /**
     * @brief Compares the current type with another type.
     *
     * The comparisons method compares the current type with another type and
     * returns a TypeComparisons object which provides information about the
     * comparison results.
     *
     * @param other_type A pointer to the other type to compare with.
     *
     * @return A TypeComparisons object containing information about the
     * comparison results.
     */
    RPY_NO_DISCARD const type_support::TypeComparisons&
    comparisons(const Type* other_type) const;
    /**
     * @brief Returns the conversions for a given type.
     *
     * The `conversions` method returns the conversions for a given type. It
     * takes a pointer to another `Type` object and returns a reference to a
     * `TypeConversions` object that contains the conversions from the current
     * type to the other type.
     *
     * @param other_type A pointer to the other `Type` object for which the
     *                   conversions are requested.
     *
     * @return A reference to the `TypeConversions` object that contains the
     *         conversions from the current type to the other type.
     */
    RPY_NO_DISCARD const type_support::TypeConversions&
    conversions(const Type* other_type) const;

    /**
     * @brief Updates the support for a given Type.
     *
     * This method updates the support for the given Type by calling the
     * `get_mut_implementor` method of the `TypeSupport` class. It returns a
     * `GuardedRef` object that provides a guarded reference to the updated
     * `TypeSupport` object, ensuring thread safety with the provided mutex.
     *
     * @param other The pointer to the Type object for which the support needs
     * to be updated.
     *
     * @return A `GuardedRef` object that provides a guarded reference to the
     * updated `TypeSupport` object.
     */
    RPY_NO_DISCARD GuardedRef<TypeSupport, std::mutex>
    update_support(const Type* other) const;

    /**
     * @brief Display the value of a type.
     *
     * The display method is used to display the value of a type. It outputs the
     * value to the specified output stream.
     *
     * @param os The output stream to display the value.
     * @param ptr A pointer to the value of the type.
     *
     * @note This method is intended to be called by the Type class.
     */
    virtual void display(std::ostream& os, const void* ptr) const;

    const type_support::NumTraits* num_traits() const noexcept
    {
        return p_num_traits.get();
    }

    /**
     * @brief Returns the zero element for the type.
     *
     * The zero method returns the zero element for the type. If the type does
     * not have a zero element, a runtime_error exception is thrown.
     *
     * @return The zero element for the type.
     *
     * @throws std::runtime_error If the type does not have a zero element.
     */
    RPY_NO_DISCARD virtual ConstReference zero() const;
    /**
     * @brief Retrieve the one element of the Type.
     *
     * This function returns the one element of the Type.
     * If the Type does not have a one element, an exception of type
     * std::runtime_error is thrown.
     *
     * @return The one element of the Type.
     *
     * @throws std::runtime_error if the Type does not have a one element.
     */
    RPY_NO_DISCARD virtual ConstReference one() const;
    /**
     * @brief Returns a constant reference to the minus one element of the type.
     *
     * The mone() method returns a constant reference to the minus one element
     * of the type. If the type does not have a one element, an exception of
     * type std::runtime_error is thrown.
     *
     * @return A constant reference to the one element of the type.
     * @throws std::runtime_error if the type does not have a minus one element.
     */
    RPY_NO_DISCARD virtual ConstReference mone() const;
};

/**
 * @brief The void_type struct.
 *
 * The void_type struct represents the type `void`. The `void` type is an empty
 * type that is used to indicate the absence of value or a function that does
 * not return a value. This struct is mainly used as a placeholder or when
 * specializing templates.
 */
RPY_NO_DISCARD ROUGHPY_DEVICES_EXPORT TypePtr void_type() noexcept;


/**
 * @brief Get the type traits for a given type.
 *
 * This function returns an instance of the TypeTraits struct that provides
 * information about the traits of a given type.
 *
 * @tparam T The type to get the traits for.
 * @return TypeTraits An instance of TypeTraits struct representing the traits
 * of the given type T.
 *
 * @note The function is declared as constexpr and is noexcept, ensuring it can
 * be used in compile-time evaluations and does not throw any exceptions.
 */
template <typename T>
constexpr TypeTraits traits_of() noexcept
{
    using base_t = remove_cv_ref_t<T>;
    return {
            is_standard_layout_v<base_t>,
            is_trivially_copyable_v<base_t>,
            is_trivially_constructible_v<base_t>,
            is_trivially_default_constructible_v<base_t>,
            is_trivially_copy_constructible_v<base_t>,
            is_trivially_copy_assignable_v<base_t>,
            is_trivially_destructible_v<base_t>,
            is_polymorphic_v<base_t>,
            is_signed_v<base_t>,
            is_floating_point_v<base_t>,
            is_integral_v<base_t>,
    };
}

/**
 * @brief Get the size in bytes of a given type.
 *
 * The `size_of` function returns the size in bytes of the specified type.
 *
 * @param type A pointer to the Type object representing the type.
 *
 * @return The size in bytes of the specified type.
 *
 * @note The `type` parameter should not be `nullptr`. The behavior is
 * undefined if `type` is `nullptr`.
 *
 * @note The returned value includes any padding bytes that may exist
 * in the type's memory layout.
 *
 * @note The returned value is based on the `bytes` field of the TypeInfo
 * object associated with the type.
 *
 * @note The returned value is not dependent on the number of instances
 * of the type, but rather on the size of a single instance of the type.
 */
inline dimn_t size_of(const Type* type) { return type->bytes(); }

/**
 * @brief Get the alignment of a given type.
 *
 * This function returns the minimum alignment requirement of a given type. The
 * alignment of a type is the byte offset at which an object must be allocated
 * in memory.
 *
 * @param type Pointer to the type for which the alignment needs to be
 * determined.
 *
 * @return The alignment of the given type.
 */
inline dimn_t align_of(const Type* type) { return type->min_alignment(); }

namespace traits {

using TypePtr = const Type*;

/**
 * @brief Check if the given type has standard layout
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type has standard layout
 */
inline bool is_standard_layout(TypePtr const typePtr)
{
    return typePtr->type_traits().standard_layout;
}

/**
 * @brief Check if the given type is trivially copyable
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is trivially copyable
 */
inline bool is_trivially_copyable(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_copyable;
}

/**
 * @brief Check if the given type is trivially constructible
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is trivially
 * constructible
 */
inline bool is_trivially_constructible(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_constructible;
}

/**
 * @brief Check if the given type is trivially default constructible
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is trivially default
 * constructible
 */
inline bool is_trivially_default_constructible(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_default_constructible;
}

/**
 * @brief Check if the given type is trivially copy constructible
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is trivially copy
 * constructible
 */
inline bool is_trivially_copy_constructible(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_copy_constructible;
}

/**
 * @brief Check if the given type is trivially copy assignable
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is trivially copy
 * assignable
 */
inline bool is_trivially_copy_assignable(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_copy_assignable;
}

/**
 * @brief Check if the given type is trivially destructible
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is trivially destructible
 */
inline bool is_trivially_destructible(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_destructible;
}

/**
 * @brief Check if the given type is polymorphic
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is polymorphic
 */
inline bool is_polymorphic(TypePtr const typePtr)
{
    return typePtr->type_traits().polymorphic;
}

/**
 * @brief Check if the given type is signed
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is signed
 */
inline bool is_signed(TypePtr const typePtr)
{
    return typePtr->type_traits().is_signed;
}

/**
 * @brief Check if the given type is unsigned
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is unsigned
 */
inline bool is_unsigned(TypePtr const typePtr)
{
    return !typePtr->type_traits().is_signed;
}

/**
 * @brief Check if the given type is floating point
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is floating point
 */
inline bool is_floating_point(TypePtr const typePtr)
{
    return typePtr->type_traits().is_floating_point;
}

/**
 * @brief Check if the given type is integral
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is integral
 */
inline bool is_integral(TypePtr const typePtr)
{
    return typePtr->type_traits().is_integral;
}

/**
 * @brief Check if the given type is either integral or floating point
 * @param typePtr A pointer to the Type object
 * @return A boolean value indicating whether the type is either integral or
 * floating point
 */
inline bool is_arithmetic(TypePtr const typePtr)
{
    return is_integral(typePtr) || is_floating_point(typePtr);
}

}// namespace traits

/**
 *
 */
ROUGHPY_DEVICES_EXPORT void register_type(const Type* tp);

/**
 *
 */
RPY_NO_DISCARD ROUGHPY_DEVICES_EXPORT const Type* get_type(string_view type_id);

/**
 * @brief Retrieves the Type for a given TypeInfo object.
 *
 * The get_type function returns the Type for a given TypeInfo object based on
 * the provided information.
 *
 * @param info The TypeInfo object that contains information about the desired
 * Type.
 * @return The Type object corresponding to the provided TypeInfo.
 * @throws std::runtime_error if the TypeInfo does not correspond to a
 * fundamental type.
 *
 * The get_type function uses a switch statement to handle different TypeCode
 * values and their corresponding byte sizes in order to return the appropriate
 * Type object. If the TypeInfo does not correspond to a fundamental type, an
 * exception of type std::runtime_error is thrown.
 */
RPY_NO_DISCARD ROUGHPY_DEVICES_EXPORT const Type*
get_type(devices::TypeInfo info);

// #ifndef RPY_NO_RTTI
//
// ROUGHPY_DEVICES_EXPORT
// void register_type(const std::type_info& info, const Type* type);
//
//
// /**
//  * @brief Get the Type object associated with the given type_info.
//  *
//  * This function retrieves the Type object associated with the given
//  type_info.
//  * The Type object provides information about various traits of the type.
//  *
//  * @param info The type_info object representing the type.
//  * @return The Type object associated with the type.
//  */
// RPY_NO_DISCARD ROUGHPY_DEVICES_EXPORT const Type*
// get_type(const std::type_info& info);
//
// template <typename T>
// const Type* get_type()
// {
//     return get_type(typeid(T));
// }
// #else

/*
 * The get_type template function is a nice way to get the abstract RoughPy type
 * from a concrete C++ type. The intention is that the implementor will provide
 * an override of type_id_of_impl<T> with the value that matches the id of the
 * abstract type in the type index. (The id field of the Type.) Alternatively,
 * if the library actually provides the C++ type, one can provide a exported
 * specialisation of this template that short-circuits the call to the type
 * cache.
 */

template <typename T>
const Type* get_type()
{
    return get_type(type_id_of<T>);
}

template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<int8_t>();

// Specializations for int16_t, int32_t, and int64_t
template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<int16_t>();

template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<int32_t>();

template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<int64_t>();

// Specializations for uint8_t, uint16_t, uint32_t, and uint64_t
template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<uint8_t>();

template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<uint16_t>();

template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<uint32_t>();

template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<uint64_t>();

// Specializations for float and double
template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<float>();

template <>
ROUGHPY_DEVICES_EXPORT RPY_NO_DISCARD const Type* get_type<double>();

// #endif

namespace dtl {

RPY_NO_DISCARD inline const Type*
compute_promotion(const Type* left, const Type* right)
{
    if (left == nullptr) { return right; }
    if (right == nullptr) { return left; }
    switch (left->compare_with(right)) {
        case TypeComparison::AreSame:
        case TypeComparison::TriviallyConvertible:
        case TypeComparison::Convertible: return left;
        case TypeComparison::NotConvertible: break;
    }
    // If we're here, it's because left is not convertible to right (or left has
    // no knowledge of right), so check again with right.

    switch (right->compare_with(left)) {
        case TypeComparison::TriviallyConvertible:
        case TypeComparison::Convertible: return right;
        case TypeComparison::NotConvertible: break;
        case TypeComparison::AreSame:
            // This case cannot be reached since we would already have seen this
            // in the first switch statement.
            RPY_UNREACHABLE_RETURN(nullptr);
    }

    RPY_THROW(
            std::invalid_argument,
            "cannot compute a common promotion for types "
                    + string(left->name()) + " and " + string(right->name())
    );
}

}// namespace dtl

/**
 * @brief Computes the promotion of a collection of types.
 *
 * This method takes a slice of Type pointers and computes the promotion of
 * these types using the dtl::compute_promotion function. The promotion of a
 * type determines the resulting type when performing operations between
 * different types. The resulting type is the most "general" type that can
 * represent all the types in the collection.
 *
 * @param types A slice of Type pointers representing the types to compute
 *              promotion for.
 * @return The promotion of the types in the collection as a const Type pointer.
 */
RPY_NO_DISCARD inline const Type* compute_promotion(Slice<const Type*> types)
{
    return ranges::fold_left(types, nullptr, dtl::compute_promotion);
}

}// namespace devices

}// namespace rpy

#endif// ROUGHPY_DEVICES_TYPE_H
