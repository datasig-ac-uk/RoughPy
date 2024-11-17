//
// Created by sammorley on 15/11/24.
//

#ifndef ROUGHPY_GENERICS_FROM_TRAIT_H
#define ROUGHPY_GENERICS_FROM_TRAIT_H

#include <memory>

#include "roughpy/core/macros.h"
#include "roughpy/core/traits.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "roughpy/core/debug_assertion.h"
#include "type_ptr.h"

#include <random>

namespace rpy::generics {

class ConstRef;
class Ref;
class Value;

class ROUGHPY_PLATFORM_EXPORT ConversionTrait
{
    TypePtr p_src_type;
    TypePtr p_dst_type;

protected:
    ConversionTrait(const Type* src_type, const Type* dst_type)
        : p_src_type(src_type),
          p_dst_type(dst_type)
    {}

public:

    RPY_NO_DISCARD
    const TypePtr& src_type() const noexcept { return p_src_type; }
    RPY_NO_DISCARD
    const TypePtr& dst_type() const noexcept { return p_dst_type; }

    virtual ~ConversionTrait();

    virtual void unsafe_convert(void* dst, const void* src) const = 0;

    void convert(Ref dst, ConstRef src) const;

    RPY_NO_DISCARD
    Value convert(ConstRef src) const;
};





template <typename From, typename To>
class RPY_LOCAL ConversionTraitImpl : public ConversionTrait
{
    static_assert(is_convertible_v<From, To>, "From must be convertible to To");

public:
    ConversionTraitImpl(const Type* from_type, const Type* to_type)
        : ConversionTrait(from_type, to_type)
    {}

    void unsafe_convert(void* dst, const void* src) const override;
};

template <typename From, typename To>
void ConversionTraitImpl<From, To>::unsafe_convert(void* dst, const void* src)
        const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);
    const auto* src_obj = static_cast<const From*>(src);
    *static_cast<const To*>(dst) = static_cast<To>(src_obj);
}



}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_FROM_TRAIT_H
