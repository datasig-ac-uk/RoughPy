//
// Created by sammorley on 15/11/24.
//

#ifndef ROUGHPY_GENERICS_FROM_TRAIT_H
#define ROUGHPY_GENERICS_FROM_TRAIT_H

#include <limits>
#include <memory>

#include "roughpy/core/macros.h"
#include "roughpy/core/traits.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "roughpy/core/debug_assertion.h"
#include "type_ptr.h"

#include <random>
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
          p_dst_type(std::move(dst_type))
    {}

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

namespace dtl {

template <typename From, typename To>
inline constexpr bool exact_convertible_to_floating_v
        = (is_floating_point_v<From> && sizeof(From) <= sizeof(To))
        || (is_integral_v<From>
            && (std::numeric_limits<From>::digits
                <= std::numeric_limits<To>::digits));

template <typename From, typename To>
inline constexpr bool exact_convertible_to_integer_v
        = (is_integral_v<From> && is_signed_v<From> == is_signed_v<To>
           && sizeof(From) <= sizeof(To));

}// namespace dtl

template <typename From, typename To>
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
class ConversionTraitImpl : public ConversionTrait
{
    static_assert(is_convertible_v<From, To>, "From must be convertible to To");

    static constexpr bool conversion_is_exact
            = (is_floating_point_v<To>
               && dtl::exact_convertible_to_floating_v<From, To>)
            || (is_integral_v<To>
                && dtl::exact_convertible_to_integer_v<From, To>);

public:
    ConversionTraitImpl(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    bool is_exact() const noexcept override;
    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

template <typename From, typename To>
bool ConversionTraitImpl<From, To>::is_exact() const noexcept
{
    return conversion_is_exact;
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

    if constexpr (!conversion_is_exact) {
        if (exact) {
            // If the conversion is not always exact, we might want to check.
            // The easiest way is to do a round trip and compare the end product
            From check = static_cast<From>(*static_cast<const To*>(dst));
            RPY_CHECK_EQ(check, *src_obj);
        }
    }
}

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_FROM_TRAIT_H
