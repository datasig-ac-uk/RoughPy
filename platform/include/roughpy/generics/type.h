//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_GENERICS_TYPE_H
#define ROUGHPY_GENERICS_TYPE_H

#include <memory>
#include <typeinfo>

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/reference_counting.h>
#include <roughpy/platform/roughpy_platform_export.h>

#include "builtin_trait.h"
#include "conversion_trait.h"
#include "roughpy/core/hash.h"
#include "type_ptr.h"

namespace rpy::generics {

/**
 * @brief Represents basic properties and functionalities within the
 * application.
 *
 * The BasicProperties class provides a foundation for various properties and
 * functionalities that are needed across different components of the system. It
 * includes methods and attributes which can be leveraged to manage and
 * manipulate these properties efficiently and effectively.
 */
struct BasicProperties
{
    bool standard_layout: 1;
    bool trivially_copyable: 1;
    bool trivially_constructible: 1;
    bool trivially_default_constructible: 1;
    bool trivially_copy_constructible: 1;
    bool trivially_copy_assignable: 1;
    bool trivially_destructible: 1;
    bool polymorphic: 1;
    bool is_signed: 1;
    bool is_floating_point: 1;
    bool is_integral: 1;
};

template <typename T>
/**
 * @brief Determines the basic properties of a given type.
 *
 * This function evaluates the provided type and returns a BasicProperties
 * object encapsulating various properties of the type, such as whether it's
 * trivially copyable, trivially destructible, or polymorphic.
 *
 * @tparam T The type for which the properties are to be determined.
 * @return A BasicProperties object containing information about the type.
 */
constexpr BasicProperties basic_properties_of() noexcept;


template <typename T>
TypePtr get_type() noexcept;

//{
//    static_assert(false, "There is no Type associated with T");
//    RPY_UNREACHABLE_RETURN(nullptr);
//}

/**
 * @brief Encapsulates the definition and functionalities related to custom
 * types.
 *
 * The Type class is used to define, represent, and manipulate various custom
 * types within the application. It provides methods and properties to interact
 * with these types, ensuring they are correctly managed throughout the system.
 */
class ROUGHPY_PLATFORM_EXPORT Type : public mem::PolymorphicRefCounted
{
    friend class Value;

public:
    /**
     * @brief Returns the type information of the current instance.
     *
     * This function retrieves the stored type information for the current
     * instance of the Type class.
     *
     * @return A reference to a std::type_info object that represents the type
     * information.
     */
    RPY_NO_DISCARD virtual const std::type_info& type_info() const noexcept = 0;

    /**
     * @brief Retrieves the basic properties of the Type instance.
     *
     * This function provides access to the underlying BasicProperties structure
     * associated with the Type instance. It allows inspection of various
     * fundamental characteristics pertinent to the type.
     *
     * @return A const reference to the BasicProperties structure.
     */
    RPY_NO_DISCARD virtual BasicProperties
    basic_properties() const noexcept = 0;

    /**
     * @brief Retrieves the object size for the Type instance.
     *
     * This function returns the size of the object that the Type instance
     * represents. The size is determined at the instance creation and remains
     * constant throughout the lifetime of the Type.
     *
     * @return The size, in bytes, of the object that the Type instance
     * represents.
     */
    RPY_NO_DISCARD virtual size_t object_size() const noexcept = 0;

    /**
     * @brief Retrieves the name associated with the current instance.
     *
     * This function returns a string view representing the name of the specific
     * instance. It provides a constant time operation to access the instance's
     * name as a non-owning view.
     *
     * @return A string_view object representing the name of the instance.
     */
    RPY_NO_DISCARD virtual string_view name() const noexcept = 0;

    /**
     * @brief Retrieves the unique identifier for the current instance.
     *
     * This function returns a string view that uniquely identifies the current
     * instance of the class. It provides a constant time operation to access
     * the instance's identifier in a non-owning view.
     *
     * @return A string_view object representing the unique identifier of the
     * instance.
     */
    RPY_NO_DISCARD virtual string_view id() const noexcept = 0;

protected:
    /**
     * @brief Allocates an instance of the represented type.
     *
     * This pure virtual function is responsible for allocating an object of the
     * type that this Type instance represents. The function returns a pointer
     * to the newly allocated object. Subclasses need to provide the
     * implementation to specify how the actual allocation is performed.
     *
     * @return A void pointer to the newly allocated object.
     */
    virtual void* allocate_object() const = 0;

    /**
     * @brief Frees an instance of the represented type.
     *
     * This pure virtual function is responsible for deallocating an object of
     * the type that this Type instance represents. The function must be
     * implemented by subclasses to define the specific deallocation procedure.
     * The function ensures that resources occupied by the object are properly
     * released.
     *
     * @param ptr A void pointer to the object that needs to be deallocated.
     */
    virtual void free_object(void* ptr) const = 0;

public:
    virtual bool parse_from_string(void* data, string_view str) const noexcept;

    /**
     * @brief Abstract function for copying or moving a block of memory.
     *
     * This pure virtual function is intended for copying or moving a specified
     * number of bytes of data from a source to a destination. The operation to
     * be performed (copy or move) depends on the boolean flag provided.
     *
     * @param dst Pointer to the destination memory location.
     * @param src Pointer to the source memory location.
     * @param count Number of bytes to be copied or moved.
     * @param uninit Flag indicating the operation type:
     *             - true if the data should be assumed uninitialized
     *             - false if the data is already initialised
     */
    virtual void
    copy_or_fill(void* dst, const void* src, size_t count, bool uninit) const
    = 0;


    virtual void move(void* dst, void* src, size_t count, bool uninit) const;

    virtual void
    destroy_range(void* data, size_t count) const = 0;


    /**
     * @brief Converts the current Type to another specified Type.
     *
     * This method facilitates the conversion of the current Type instance to
     * another Type instance as specified by the type parameter.
     *
     * @param type The target Type to convert to.
     * @return A unique pointer to a constant ConversionTrait representing the
     *         conversion result.
     */
    RPY_NO_DISCARD virtual std::unique_ptr<const ConversionTrait>
    convert_to(const Type& type) const noexcept;

    /**
     * @brief Converts the given custom type to another type using its
     * conversion traits.
     *
     * This method provides a way to convert from one Type object to another by
     * utilizing the associated conversion traits. The conversion ensures that
     * the resulting type adheres to the expected characteristics defined by the
     * conversion traits.
     *
     * @param type The Type object that needs to be converted.
     * @return A unique pointer to the resulting ConversionTrait after
     * conversion.
     */
    RPY_NO_DISCARD virtual std::unique_ptr<const ConversionTrait>
    convert_from(const Type& type) const noexcept;

    /**
     * @brief Retrieves the built-in trait associated with the given ID.
     *
     * This method returns the built-in trait corresponding to the specified
     * BuiltinTraitID. It provides a mechanism to access built-in trait
     * functionalities based on their unique identifiers.
     *
     * @param id The identifier of the built-in trait to be retrieved.
     * @return A pointer to the BuiltinTrait corresponding to the given ID,
     * or nullptr if the trait is not found.
     */
    RPY_NO_DISCARD virtual const BuiltinTrait*
    get_builtin_trait(BuiltinTraitID id) const noexcept;

    // RPY_NO_DISCARD virtual const Trait* get_trait(string_view id
    // ) const noexcept;

    /**
     * @brief Displays the value pointed to by the given pointer using the
     * provided output stream.
     *
     * This pure virtual function is intended to be overridden by derived
     * classes to provide a specific implementation for displaying values of a
     * custom type.
     *
     * @param os The output stream used for displaying the value.
     * @param value A pointer to the value that needs to be displayed.
     * @return A reference to the output stream after the value has been written
     * to it.
     */
    virtual const std::ostream&
    display(std::ostream& os, const void* value) const = 0;

    /**
     * @brief Computes the hash value for a given input.
     *
     * This function takes a pointer to an input value and returns its computed
     * hash. It ensures that the hash calculation is performed without throwing
     * any exceptions.
     *
     * @param value A pointer to the input value whose hash is to be computed.
     * @return The hash value of the input.
     */
    virtual hash_t hash_of(const void* value) const noexcept;


    template <typename T>
    /**
     * @brief Retrieves the type information for a specified type.
     *
     * This method provides a type-safe way to obtain the TypePtr for the given
     * type, utilizing the decay_t transformation to handle type decay.
     *
     * @return A TypePtr representing the specific type.
     */
    static TypePtr of() noexcept { return get_type<decay_t<T> >(); }

};

/**
 * @brief Represents a collection of built-in types.
 *
 * The BuiltinTypes struct encapsulates various fundamental built-in types
 * such as floating-point types and integer types.
 */
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
};

/**
 * @brief Retrieves the set of built-in types.
 *
 * @return A reference to a static BuiltinTypes object containing all built-in
 * types.
 */
ROUGHPY_PLATFORM_EXPORT
const BuiltinTypes& get_builtin_types() noexcept;


class ROUGHPY_PLATFORM_EXPORT MultiPrecisionTypes
{
    MultiPrecisionTypes();

public:
    TypePtr integer_type;
    TypePtr rational_type;

    RPY_NO_DISCARD
    TypePtr float_type(int n_precision) const;

    RPY_NO_DISCARD
    static const MultiPrecisionTypes& get() noexcept;
};

RPY_NO_DISCARD
ROUGHPY_PLATFORM_EXPORT
TypePtr get_polynomial_type() noexcept;


template <typename T>
/**
 * @brief Determines the basic properties of a given type.
 * @tparam T The type for which the properties are to be determined.
 * @return A BasicProperties object containing information about the type.
 */
constexpr BasicProperties basic_properties_of() noexcept
{
    using base_t = remove_cvref_t<T>;
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
 * @brief Computes the type promotion for two given types.
 *
 * The function determines the resulting type when the two provided types
 * interact according to the promotion rules. If either of the types is
 * `nullptr`, the other type is returned. If both types are the same,
 * that type is returned. Otherwise, the function checks if one type
 * is exactly convertible to the other and returns the appropriate type.
 *
 * @param lhs A pointer to the first type.
 * @param rhs A pointer to the second type.
 * @return The promoted type based on the provided `lhs` and `rhs` types.
 * If neither type can be promoted to the other, `nullptr` is returned.
 */
RPY_NO_DISCARD TypePtr ROUGHPY_PLATFORM_EXPORT compute_promotion(
    const Type* lhs,
    const Type* rhs) noexcept;

/**
 * @brief Computes the hash value of a given Type object.
 *
 * This function creates a hash value for the provided Type object by invoking
 * the hash function on the type's identifier string. It uses the
 * `Hash<string_view>` hasher to generate the hash.
 *
 * @param value A reference to a Type object whose hash value needs to be
 * computed.
 * @return The computed hash value.
 */
inline hash_t hash_value(const Type& value) noexcept
{
    constexpr Hash<string_view> hasher;
    return hasher(value.id());
}

RPY_NO_DISCARD constexpr bool
operator==(const Type& lhs, const Type& rhs) noexcept { return &lhs == &rhs; }

RPY_NO_DISCARD constexpr bool
operator!=(const Type& lhs, const Type& rhs) noexcept { return !(lhs == rhs); }

/**
 * @brief Retrieves the size of an object of the specified type.
 *
 * This function returns the size of an object for the given type by calling
 * the `object_size` method of the `Type` class.
 *
 * @param type A reference to the `Type` object for which the size is required.
 * @return The size of the object for the given type.
 */
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