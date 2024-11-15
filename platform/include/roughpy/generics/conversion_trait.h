//
// Created by sammorley on 15/11/24.
//

#ifndef ROUGHPY_GENERICS_FROM_TRAIT_H
#define ROUGHPY_GENERICS_FROM_TRAIT_H

#include <memory>

#include "roughpy/core/macros.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "type_ptr.h"

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

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_FROM_TRAIT_H
