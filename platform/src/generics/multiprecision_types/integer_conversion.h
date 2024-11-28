//
// Created by sam on 28/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_INTEGER_CONVERSION_H
#define ROUGHPY_GENERICS_INTERNAL_INTEGER_CONVERSION_H

#include <memory>

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/generics/type_ptr.h"
#include "roughpy/generics/conversion_trait.h"

namespace rpy {
namespace generics {

class MPIntegerConversionFromFactory {
public:
    virtual ~MPIntegerConversionFromFactory() = default;

    RPY_NO_DISCARD virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;

    static const MPIntegerConversionFromFactory* get_factory(const Type& type);
};


class MPIntegerConversionToFactory {
public:
    virtual ~MPIntegerConversionToFactory() = default;

    RPY_NO_DISCARD virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;


    static const MPIntegerConversionToFactory* get_factory(const Type& type);
};

} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_INTEGER_CONVERSION_H
