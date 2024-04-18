//
// Created by sam on 4/17/24.
//

#include "scalars_fwd.h"

#include "scalar/do_macro.h"
#include "scalar_implementations/arbitrary_precision_rational.h"
#include "scalar_type.h"
#include "traits.h"

using namespace rpy;
using namespace rpy::scalars;

PackedScalarType scalars::rational_type_of(PackedScalarType type)
{
    RPY_CHECK(!type.is_null());
    if (type.is_pointer()) { return type->rational_type(); }

    const auto info = type.get_type_info();
    if (info.code == devices::TypeCode::APRationalPolynomial) {
        return devices::type_info<ArbitraryPrecisionRational>();
    }

    return info;
}
