//
// Created by sam on 07/07/23.
//

#ifndef ROUGHPY_SCALARS_CONVERSION_H
#define ROUGHPY_SCALARS_CONVERSION_H

#include "scalar_pointer.h"
#include "scalar_type.h"

namespace rpy {
namespace scalars {

template <typename T>
RPY_NO_DISCARD inline enable_if_t<is_default_constructible<T>::value, T>
try_convert(ScalarPointer arg, const ScalarType* type = nullptr)
{
    if (type == nullptr) { type = ScalarType::of<T>(); }

    if (arg.is_null()) { return T(0); }
    const auto* arg_type = arg.type();
    if (arg_type == nullptr) {
        RPY_THROW(std::runtime_error, "null type for non-zero value");
    }

    auto cv = get_conversion(arg_type->id(), type->id());
    if (!cv) {
        RPY_THROW(std::runtime_error,
                "no known conversion from " + arg_type->info().name
                + " to type " + type->info().name
        );
    }
    T result;
    cv({type, &result}, arg, 1);
    return result;
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_CONVERSION_H
