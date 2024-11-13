//
// Created by sam on 13/11/24.
//

#include "roughpy/generics/traits/from_trait.h"


#include "roughpy/generics/const_reference.h"
#include "roughpy/generics/reference.h"
#include "roughpy/generics/type.h"
#include "roughpy/generics/value.h"


using namespace rpy;
using namespace rpy::generics;


FromTrait::~FromTrait() = default;



void FromTrait::from(Reference dst, ConstReference src) const
{

}
Value FromTrait::from(ConstReference src) const
{
    Value result(p_to);

}
