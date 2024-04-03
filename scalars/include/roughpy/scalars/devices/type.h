//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_DEVICES_TYPE_H
#define ROUGHPY_DEVICES_TYPE_H

#include "core.h"

#include <mutex>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/scalars/scalars_fwd.h>

namespace rpy {
namespace devices {

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

class ROUGHPY_SCALARS_EXPORT Type
{
    string_view m_id;
    string_view m_name;
    TypeInfo m_info;
    TypeTraits m_traits;

public:
    explicit
    Type(string_view id, string_view name, TypeInfo info, TypeTraits traits);

    Type() = delete;
    Type(const Type&) = delete;
    Type(Type&&) noexcept = delete;

    Type& operator=(const Type&) = delete;
    Type& operator=(Type&&) noexcept = delete;

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

    RPY_NO_DISCARD TypeInfo type_info() const noexcept { return m_info; }

    RPY_NO_DISCARD const TypeTraits& type_traits() const noexcept
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

    virtual bool supports_device(const Device& device) const noexcept;
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

}// namespace traits

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT const devices::Type*
get_type(devices::TypeInfo info);

}// namespace devices

}// namespace rpy

#endif// ROUGHPY_DEVICES_TYPE_H
