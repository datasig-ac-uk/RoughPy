//
// Created by sammorley on 15/11/24.
//

#ifndef ROUGHPY_GENERICS_FROM_TRAIT_H
#define ROUGHPY_GENERICS_FROM_TRAIT_H


#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "type_ptr.h"


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
    ConversionTrait(TypePtr src_type, TypePtr dst_type);

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


template <typename From, typename To, typename=void>
struct ConversionHelper
{
    using from_ptr = const From*;
    using to_ptr = To*;

    static constexpr bool is_possible = false;

    // Set to true if the conversion is guaranteed to be exact
    static constexpr bool is_always_exact = false;

    // The conversion implementation. Should only check for an inexact
    // conversion if the ensure_exact flag is set (and if the is_always_exact is
    // false).
    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact);

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
template <typename From, typename To, typename SFINEA=void>
class ConversionTraitImpl : public ConversionTrait
{
    using helper = ConversionHelper<From, To>;
    static_assert(helper::is_possible, "From must be convertible to To");

public:
    ConversionTraitImpl(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type)) {}

    bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

template <typename From, typename To, typename SFINEA>
bool ConversionTraitImpl<From, To, SFINEA>::is_exact() const noexcept
{
    return helper::is_always_exact;
}

template <typename From, typename To, typename SFINEA>
void ConversionTraitImpl<From, To, SFINEA>::unsafe_convert(void* dst,
    const void* src,
    bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* dst_ptr = static_cast<typename helper::to_ptr>(dst);
    auto* src_ptr = static_cast<typename helper::from_ptr>(src);

    auto result = helper::convert(dst_ptr, src_ptr, exact);
    if (result == ConversionResult::Failed) {
        throw std::runtime_error("conversion failed");
    }
    if (exact && ConversionResult::Inexact == result) {
        RPY_THROW(std::runtime_error,
            "conversion was required to be precise but was not");
    }
}

}

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_FROM_TRAIT_H