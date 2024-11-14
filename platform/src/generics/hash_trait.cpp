//
// Created by sam on 14/11/24.
//

#include "roughpy/generics/hash_trait.h"

#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/types.h"

#include "roughpy/generics/values.h"

using namespace rpy;
using namespace rpy::generics;


 HashTrait::~HashTrait() = default;


hash_t HashTrait::hash(ConstRef value) const
{
    if (value.fast_is_zero()) {
        return static_cast<hash_t>(0);
    }

    return unsafe_hash(value.data());
}

