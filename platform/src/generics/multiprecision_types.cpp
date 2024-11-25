//
// Created by sammorley on 25/11/24.
//

#include "roughpy/generics/type.h"

#include <limits>

#include "multiprecision_types/rational_type.h"

using namespace rpy;
using namespace rpy::generics;


MultiPrecisionTypes::MultiPrecisionTypes()
    : integer_type(nullptr),
      rational_type(RationalType::get())
{

}

TypePtr MultiPrecisionTypes::float_type(size_t n_precision) const {
    if (n_precision <= std::numeric_limits<float>::digits) {
        return get_builtin_types().float_type;
    }
    if (n_precision <= std::numeric_limits<double>::digits) {
        return get_builtin_types().double_type;
    }



    RPY_THROW(std::domain_error, "this precision is not available");
}

const MultiPrecisionTypes& MultiPrecisionTypes::get() noexcept
{
    static const MultiPrecisionTypes object;
    return object;
}
