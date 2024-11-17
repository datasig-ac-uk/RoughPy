//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_GENERICS_TYPE_H
#define ROUGHPY_GENERICS_TYPE_H

#include <atomic>
#include <memory>
#include <typeinfo>

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/roughpy_platform_export.h>

#include "builtin_trait.h"
#include "conversion_trait.h"
#include "roughpy/core/hash.h"
#include "type_ptr.h"

namespace rpy::generics {


struct BasicProperties {
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

template <typename T>
constexpr BasicProperties basic_properties_of() noexcept;


template <typename T>
TypePtr get_type() noexcept
{
    static_assert(false, "There is no Type associated with T");
    RPY_UNREACHABLE_RETURN(nullptr);
}


class ROUGHPY_PLATFORM_EXPORT Type
{
    mutable std::atomic_intptr_t m_rc;
    const std::type_info* p_type_info;

    size_t m_obj_size;
    BasicProperties m_basic_properties;

    friend class Value;

protected:
    explicit
    Type(const std::type_info* real_type_info,
         size_t obj_size,
         BasicProperties properties)
        : m_rc(1),
          p_type_info(real_type_info),
          m_obj_size(obj_size),
          m_basic_properties(properties)
    {}

    virtual ~Type();

    virtual void inc_ref() const noexcept;
    virtual bool dec_ref() const noexcept;

public:

    intptr_t ref_count() const noexcept;

    friend void intrusive_ptr_add_ref(const Type* value) noexcept
    {
        value->inc_ref();
    }

    friend void intrusive_ptr_release(const Type* value) noexcept
    {
        if (RPY_UNLIKELY(value->dec_ref())) { delete value; }
    }

public:
    RPY_NO_DISCARD const std::type_info& type_info() const noexcept
    {
        return *p_type_info;
    }

    RPY_NO_DISCARD constexpr const BasicProperties&
    basic_properties() const noexcept
    {
        return m_basic_properties;
    }

    RPY_NO_DISCARD constexpr size_t object_size() const noexcept
    {
        return m_obj_size;
    }

    RPY_NO_DISCARD virtual string_view name() const noexcept = 0;
    RPY_NO_DISCARD virtual string_view id() const noexcept = 0;

protected:
    virtual void* allocate_object() const = 0;
    virtual void free_object(void*) const = 0;

public:
    virtual void
    copy_or_move(void* dst, const void* src, size_t count, bool move) const noexcept
            = 0;

    RPY_NO_DISCARD virtual std::unique_ptr<const ConversionTrait>
    convert_to(const Type& type) const noexcept;

    RPY_NO_DISCARD virtual std::unique_ptr<const ConversionTrait>
    convert_from(const Type& type) const noexcept;

    RPY_NO_DISCARD virtual const BuiltinTrait*
    get_builtin_trait(BuiltinTraitID id) const noexcept;

    // RPY_NO_DISCARD virtual const Trait* get_trait(string_view id
    // ) const noexcept;


    virtual const std::ostream&
    display(std::ostream& os, const void* value) const = 0;


    template <typename T>
    static TypePtr of() noexcept
    {
        return get_type<decay_t<T>>();
    }

};




struct BuiltinTypes
{
    TypePtr float_type;
    TypePtr double_type;
    
    TypePtr int8_type;
    TypePtr uint8_type;
    TypePtr int16_type;
    TypePtr uint16_type;
    TypePtr int32_type;
    TypePtr uint32_type;
    TypePtr int64_type;
    TypePtr uint64_type;
    
    TypePtr rational_type;
    TypePtr poly_rational_type;
};

ROUGHPY_PLATFORM_EXPORT
const BuiltinTypes& get_builtin_types() noexcept;


template <typename T>
constexpr BasicProperties basic_properties_of() noexcept
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

ROUGHPY_PLATFORM_EXPORT
RPY_NO_DISCARD TypePtr
compute_promotion(const Type* lhs, const Type* rhs) noexcept;


inline hash_t hash_value(const Type& value) noexcept
{
    const Hash<string_view> hasher;
    return hasher(value.id());
}

RPY_NO_DISCARD constexpr bool
operator==(const Type& lhs, const Type& rhs) noexcept
{
    return &lhs == &rhs;
}

RPY_NO_DISCARD constexpr bool
operator!=(const Type& lhs, const Type& rhs) noexcept
{
    return !(lhs == rhs);
}

inline size_t size_of(const Type& type) noexcept { return type.object_size(); }

namespace concepts {



/**
 * @brief Check if the given type has standard layout
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type has standard layout
 */
inline bool is_standard_layout(Type const& type)
{
    return type.basic_properties().standard_layout;
}

/**
 * @brief Check if the given type is trivially copyable
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is trivially copyable
 */
inline bool is_trivially_copyable(Type const& type)
{
    return type.basic_properties().trivially_copyable;
}

/**
 * @brief Check if the given type is trivially constructible
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is trivially
 * constructible
 */
inline bool is_trivially_constructible(Type const& type)
{
    return type.basic_properties().trivially_constructible;
}

/**
 * @brief Check if the given type is trivially default constructible
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is trivially default
 * constructible
 */
inline bool is_trivially_default_constructible(Type const& type)
{
    return type.basic_properties().trivially_default_constructible;
}

/**
 * @brief Check if the given type is trivially copy constructible
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is trivially copy
 * constructible
 */
inline bool is_trivially_copy_constructible(Type const& type)
{
    return type.basic_properties().trivially_copy_constructible;
}

/**
 * @brief Check if the given type is trivially copy assignable
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is trivially copy
 * assignable
 */
inline bool is_trivially_copy_assignable(Type const& type)
{
    return type.basic_properties().trivially_copy_assignable;
}

/**
 * @brief Check if the given type is trivially destructible
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is trivially destructible
 */
inline bool is_trivially_destructible(Type const& type)
{
    return type.basic_properties().trivially_destructible;
}

/**
 * @brief Check if the given type is polymorphic
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is polymorphic
 */
inline bool is_polymorphic(Type const& type)
{
    return type.basic_properties().polymorphic;
}

/**
 * @brief Check if the given type is signed
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is signed
 */
inline bool is_signed(Type const& type)
{
    return type.basic_properties().is_signed;
}

/**
 * @brief Check if the given type is unsigned
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is unsigned
 */
inline bool is_unsigned(Type const& type)
{
    return !type.basic_properties().is_signed;
}

/**
 * @brief Check if the given type is floating point
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is floating point
 */
inline bool is_floating_point(Type const& type)
{
    return type.basic_properties().is_floating_point;
}

/**
 * @brief Check if the given type is integral
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is integral
 */
inline bool is_integral(Type const& type)
{
    return type.basic_properties().is_integral;
}

/**
 * @brief Check if the given type is either integral or floating point
 * @param type A reference to the Type object
 * @return A boolean value indicating whether the type is either integral or
 * floating point
 */
inline bool is_arithmetic(Type const& type)
{
    return is_integral(type) || is_floating_point(type);
}

}// namespace concepts



template <>
inline TypePtr get_type<float>() noexcept
{
    return get_builtin_types().float_type;
}

template <>
inline TypePtr get_type<double>() noexcept
{
    return get_builtin_types().double_type;
}

template <>
inline TypePtr get_type<int8_t>() noexcept
{
    return get_builtin_types().int8_type;
}

template <>
inline TypePtr get_type<uint8_t>() noexcept
{
    return get_builtin_types().uint8_type;
}

template <>
inline TypePtr get_type<int16_t>() noexcept
{
    return get_builtin_types().int16_type;
}

template <>
inline TypePtr get_type<uint16_t>() noexcept
{
    return get_builtin_types().uint16_type;
}

template <>
inline TypePtr get_type<int32_t>() noexcept
{
    return get_builtin_types().int32_type;
}

template <>
inline TypePtr get_type<uint32_t>() noexcept
{
    return get_builtin_types().uint32_type;
}

template <>
inline TypePtr get_type<int64_t>() noexcept
{
    return get_builtin_types().int64_type;
}

template <>
inline TypePtr get_type<uint64_t>() noexcept
{
    return get_builtin_types().uint64_type;
}


}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_TYPE_H
