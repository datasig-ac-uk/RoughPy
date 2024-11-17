//
// Created by sammorley on 15/11/24.
//


#include "roughpy/generics/conversion_trait.h"


#include <memory>

#include "roughpy/generics/type.h"
#include "roughpy/generics/values.h"


using namespace rpy;
using namespace rpy::generics;

ConversionTrait::~ConversionTrait() = default;

void ConversionTrait::convert(Ref dst, ConstRef src) const
{
    RPY_CHECK(!dst.fast_is_zero());
    RPY_CHECK_EQ(dst.type(), *p_dst_type);
    RPY_CHECK(!src.fast_is_zero());
    RPY_CHECK_EQ(src.type(), *p_src_type);

    unsafe_convert(dst.data(), src.data());
}

Value ConversionTrait::convert(ConstRef src) const
{
    RPY_CHECK(!src.fast_is_zero());
    RPY_CHECK_EQ(src.type(), *p_src_type);
    Value result(p_dst_type);
    unsafe_convert(result.data(), src.data());
    return result;
}
