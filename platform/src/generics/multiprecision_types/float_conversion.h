//
// Created by sam on 28/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_FLOAT_CONVERSION_H
#define ROUGHPY_GENERICS_INTERNAL_FLOAT_CONVERSION_H

#include <memory>

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/generics/type_ptr.h"
#include "roughpy/generics/conversion_trait.h"

namespace rpy::generics {

class MPFloatConversionFromFactory {
public:
    virtual ~MPFloatConversionFromFactory() = default;

    RPY_NO_DISCARD virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;

    static const MPFloatConversionFromFactory* get_factory(const Type& type);
};

class MPFloatConversionToFactory
{
public:
    virtual ~MPFloatConversionToFactory() = default;

    RPY_NO_DISCARD virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;

    static const MPFloatConversionToFactory* get_factory(const Type& type);
};


}

#endif //ROUGHPY_GENERICS_INTERNAL_FLOAT_CONVERSION_H
