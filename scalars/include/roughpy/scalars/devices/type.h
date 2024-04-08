//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_DEVICES_TYPE_H
#define ROUGHPY_DEVICES_TYPE_H

#include "core.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

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

/**
 * @class Type
 * @brief Represents a type with various traits and behavior.
 *
 * The Type class provides information about a type, including its name, unique
 * id, type information, and type traits. It also provides various methods for
 * working with the type, such as allocating memory for scalars, checking type
 * compatibility, and converting between types.
 */
class ROUGHPY_SCALARS_EXPORT Type
{
    string_view m_id;
    string_view m_name;
    TypeInfo m_info;
    TypeTraits m_traits;

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
    RPY_NO_DISCARD virtual string_view name() const noexcept { return m_name; }

    /**
     * @brief Get the unique internal ID string for this type
     * @return const reference to the ID string.
     */
    RPY_NO_DISCARD virtual string_view id() const noexcept { return m_id; }

    /**
     * @brief Retrieves the type information of the object.
     *
     * @return The type information of the object.
     */
    RPY_NO_DISCARD virtual TypeInfo type_info() const noexcept
    {
        return m_info;
    }

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
};

template <typename T>
constexpr TypeTraits traits_of() noexcept
{
    using base_t = remove_cv_ref_t<T>;
    return {
            is_standard_layout<base_t>::value,
            is_trivially_copyable<base_t>::value,
            std::is_trivially_constructible_v<base_t>,
            std::is_trivially_default_constructible_v<base_t>,
            std::is_trivially_copy_constructible_v<base_t>,
            std::is_trivially_copy_assignable_v<base_t>,
            std::is_trivially_destructible_v<base_t>,
            std::is_polymorphic_v<base_t>,
            std::is_signed_v<base_t>,
            std::is_floating_point_v<base_t>,
            std::is_integral_v<base_t>,
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
inline dimn_t size_of(const Type* type) { return type->type_info().bytes; }

namespace traits {

using TypePtr = const Type*;

inline bool is_standard_layout(TypePtr const typePtr)
{
    return typePtr->type_traits().standard_layout;
}

inline bool is_trivially_copyable(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_copyable;
}

inline bool is_trivially_constructible(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_constructible;
}

inline bool is_trivially_default_constructible(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_default_constructible;
}

inline bool is_trivially_copy_constructible(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_copy_constructible;
}

inline bool is_trivially_copy_assignable(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_copy_assignable;
}

inline bool is_trivially_destructible(TypePtr const typePtr)
{
    return typePtr->type_traits().trivially_destructible;
}

inline bool is_polymorphic(TypePtr const typePtr)
{
    return typePtr->type_traits().polymorphic;
}

inline bool is_signed(TypePtr const typePtr)
{
    return typePtr->type_traits().is_signed;
}

inline bool is_unsigned(TypePtr const typePtr)
{
    return !typePtr->type_traits().is_signed;
}

inline bool is_floating_point(TypePtr const typePtr)
{
    return typePtr->type_traits().is_floating_point;
}

inline bool is_integral(TypePtr const typePtr)
{
    return typePtr->type_traits().is_integral;
}

inline bool is_arithmetic(TypePtr const typePtr)
{
    return is_integral(typePtr) || is_floating_point(typePtr);
}

}// namespace traits

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT const devices::Type*
get_type(devices::TypeInfo info);

}// namespace devices

}// namespace rpy

#endif// ROUGHPY_DEVICES_TYPE_H
