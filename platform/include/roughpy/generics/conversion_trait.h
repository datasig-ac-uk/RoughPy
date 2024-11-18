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

    virtual bool is_exact() const noexcept = 0;

    virtual void unsafe_convert(void* dst, const void* src, bool exact) const
            = 0;

    void convert(Ref dst, ConstRef src, bool exact = true) const;

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
