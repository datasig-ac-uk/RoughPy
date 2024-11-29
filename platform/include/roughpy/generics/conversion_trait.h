//
// Created by sammorley on 15/11/24.
//

#ifndef ROUGHPY_GENERICS_FROM_TRAIT_H
#define ROUGHPY_GENERICS_FROM_TRAIT_H

#include <limits>
#include <memory>

#include "roughpy/core/hash.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/meta.h"
#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "roughpy/core/debug_assertion.h"
#include "type_ptr.h"

#include <random>
#include <Eigen/src/Core/Map.h>
#include <roughpy/core/check.h>

namespace rpy::generics {

class ConstRef;
class Ref;
class Value;

/**
 * @brief This class defines the interface for type conversions.
 *
 * The ConversionTrait class serves as a base class for implementing type
 * conversion operations between different types. It encapsulates the source and
 * destination types and provides methods for checking the types and performing
 * conversions.
 */
class ROUGHPY_PLATFORM_EXPORT ConversionTrait
{
    TypePtr p_src_type;
    TypePtr p_dst_type;

protected:
    ConversionTrait(TypePtr src_type, TypePtr dst_type)
        : p_src_type(std::move(src_type)),
          p_dst_type(std::move(dst_type)) {}

public:
    RPY_NO_DISCARD const TypePtr& src_type() const noexcept
    {
        return p_src_type;
    }

    RPY_NO_DISCARD const TypePtr& dst_type() const noexcept
    {
        return p_dst_type;
    }

    virtual ~ConversionTrait();

    /**
     * @brief Checks if the source type is exactly the same as the destination
     * type.
     *
     * The is_exact function compares the source type and the destination type
     * to determine if they match exactly. This can be used in type conversion
     * scenarios where an exact match is required to proceed with the operation.
     *
     * @return true if the source type and destination type are exactly the
     * same, false otherwise.
     */
    RPY_NO_DISCARD virtual bool is_exact() const noexcept = 0;

    /**
     * @brief Converts data between source and destination types without safety
     * checks.
     *
     * The unsafe_convert function performs a type conversion operation from
     * the source to the destination type specified by the implementation.
     * It does not perform any type or bounds checking, hence it is marked as
     * unsafe.
     *
     * @param dst Pointer to the destination memory where the converted data
     * will be stored.
     * @param src Pointer to the source memory from where the data will be read.
     * @param exact Flag indicating whether the conversion should be performed
     * as an exact match of types (if supported by the implementation).
     */
    virtual void unsafe_convert(void* dst, const void* src, bool exact) const
    = 0;

    /**
     * @brief Converts data from the source to the destination type.
     *
     * This method performs a type conversion operation by checking the types of
     * the source and destination, and then invoking the unsafe_convert function
     * to carry out the conversion.
     *
     * @param dst A reference to the destination object where the converted data
     * will be stored.
     * @param src A constant reference to the source object from which the data
     * will be read.
     * @param exact A boolean flag indicating whether the conversion should be
     * an exact match of types.
     */
    void convert(Ref dst, ConstRef src, bool exact = true) const;

    /**
     * @brief Converts data from the source type to the destination type with an
     * optional exact match requirement.
     *
     * This method performs a type conversion operation between the source and
     * destination types. It first ensures that the source is not zero and that
     * the source type matches the expected type. The conversion is then carried
     * out, optionally as an exact type match if specified.
     *
     * @param src A constant reference to the source object from which the data
     * will be read.
     * @param exact A boolean flag indicating whether the conversion should be
     * an exact match of types.
     * @return A Value object containing the converted data.
     */
    RPY_NO_DISCARD Value convert(ConstRef src, bool exact = true) const;
};



namespace conv {


/**
 * @brief Represents the result of a type conversion operation.
 *
 * ConversionResult is an enumeration used to indicate the outcome of
 * a conversion process, which can either be successful, inexact, or failed.
 */
enum class ConversionResult
{
    Success,
    Inexact,
    Failed
};

/**
 * @brief This class provides helper functions for data conversion operations.
 *
 * The ConversionHelpers class offers a collection of static methods designed
 * to facilitate various data conversion processes. These methods aid in
 * converting data between different formats or types, ensuring consistency
 * and correctness.
 */
template <typename T, typename U,bool Nested = false, typename SFINAE=void>
struct ConversionHelpers
{
    // conversion from U to T
    // conversion to U from T

    static constexpr bool is_nested = Nested;

    using reverse_helper = conditional_t<Nested, void, ConversionHelpers<U, T,
        true> >;

    /*
     * The primary templated just tries to use the reverse direction helper to
     * do conversions. This means we only need to specialize where for
     * combination once.
     */

    /*
     * Determine if the conversion from T to U is always exact
     */
    static constexpr bool from_exact_convertible() noexcept
    {
        if constexpr (Nested) {
            return false;
        } else {
            return reverse_helper::to_exact_convertible();
        }
    }

    /*
     * Determine if the conversion to T from U is always exact
     */
    static constexpr bool to_exact_convertible() noexcept
    {
        if constexpr (Nested) {
            return false;
        } else {
            return reverse_helper::from_exact_convertible();
        }
    }

    /*
     * Implement conversion from U to T
     * the ensure_exact parameter is used to check if the conversion was inexact
     * which is only relevant if the conversion is not infallible and not
     * guaranteed to be exact (e.g. from int32_t to int64_t is infallible and
     * always exact, but conversion from double to int64_t is not exact). Return
     * ConversionResult::Success if the conversion succeeds or if ensure_exact is false.
     * Return ConversionResult::Failed if the conversion could not be performed.
     * If ensure_exact is set true and the conversion is not always exact,
     * additionally check that the conversion result is an exact conversion,
     * returning ConversionResult::Inexact on failure.
     */
    static ConversionResult from(T* dst_ptr,
                                 const U* src_ptr,
                                 bool ensure_exact) noexcept
    {
        if constexpr (Nested) {
            return ConversionResult::Failed;
        } else {
            return reverse_helper::to(dst_ptr, src_ptr, ensure_exact);
        }
    }

    /*
     * Perform the reverse conversion from T to U. This has the same semantics as
     * the "from" method but with the roles of T and U reversed.
     */
    static ConversionResult to(U* dst_ptr,
                               const T* src_ptr,
                               bool ensure_exact) noexcept
    {
        if constexpr (Nested) {
            return ConversionResult::Failed;
        } else {
            return reverse_helper::from(dst_ptr, src_ptr, ensure_exact);
        }
    }

    // Helper to check if two values are equal
    static bool compare_equal(const T* t, const U* u) noexcept
    {
        if constexpr (Nested) {
            return false;
        } else {
            return reverse_helper::compare_equal(u, t);
        }
    }
};


/**
 * @brief Specialized implementation of the ConversionTrait for specific types.
 *
 * The ConversionTraitImpl class provides concrete functionality for type
 * conversion between specified source and destination types. It includes
 * methods for checking if the conversion is exact and performing the conversion
 * without safety checks. It ensures that the conversion between the specified
 * types is possible at compile time.
 *
 * @tparam From The source type for the conversion.
 * @tparam To The destination type for the conversion.
 */
template <typename From, typename To>
class ConversionTraitImpl : public ConversionTrait
{
    static_assert(is_convertible_v<From, To>, "From must be convertible to To");

    using helper = conv::ConversionHelpers<From, To>;

public:
    ConversionTraitImpl(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type)) {}

    bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

template <typename From, typename To>
bool ConversionTraitImpl<From, To>::is_exact() const noexcept
{
    return helper::from_exact_convertible();
}

template <typename From, typename To>
void ConversionTraitImpl<From, To>::unsafe_convert(
    void* dst,
    const void* src,
    bool exact
) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);
    const auto* src_obj = static_cast<const From*>(src);
    *static_cast<To*>(dst) = static_cast<To>(*src_obj);

    auto result = helper::From(static_cast<To*>(dst),
                               static_cast<const From*>(src),
                               exact);

    RPY_CHECK_EQ(result, conv::ConversionResult::Success);
}


/**
 * @brief This class is responsible for creating type conversion objects.
 *
 * The ConversionFactory class provides a factory mechanism to generate instances
 * of type conversion objects. It manages the creation process and simplifies
 * the instantiation of converters for various type transformations by delegating
 * the responsibility of object creation to the factory class.
 */
class ConversionFactory
{
public:
    virtual ~ConversionFactory() = default;

    RPY_NO_DISCARD
    virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;
};

template <typename From, typename To>
class ConversionFactoryImpl : public ConversionFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename From, typename To>
std::unique_ptr<const ConversionTrait> ConversionFactoryImpl<From, To>::make(
    TypePtr from_type,
    TypePtr to_type) const
{
    return std::make_unique<const ConversionTraitImpl<From, To>>(
            std::move(from_type),
            std::move(to_type)
    );
}


template <typename From, typename Map, typename To, typename... Ts>
void build_conversion_from_table(Map& map, meta::TypeList<To, Ts...> list)
{
    ignore_unused(list);
    if constexpr (!is_same_v<From, To>) {
        Hash<string_view> hasher;
        map.emplace(
            hasher(type_id_of<To>),
            std::make_unique<ConversionFactoryImpl<From, To>>()
        );
    }
    if constexpr (sizeof...(Ts) > 0) {
        build_conversion_from_table(map, meta::TypeList<Ts...>{});
    }
}



template <typename To, typename Map, typename From, typename... Ts>
void build_conversion_to_table(Map& map, meta::TypeList<From, Ts...> list)
{
    ignore_unused(list);
    if constexpr (!is_same_v<From, To>) {
        Hash<string_view> hasher;
        map.emplace(
            hasher(type_id_of<From>),
            std::make_unique<ConversionFactoryImpl<From, To>>()
        );
    }
    if constexpr (sizeof...(Ts) > 0) {
        build_conversion_from_table(map, meta::TypeList<Ts...>{});
    }
}





}// namespace conv

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_FROM_TRAIT_H