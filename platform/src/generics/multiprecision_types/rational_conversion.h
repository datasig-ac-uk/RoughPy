//
// Created by sammorley on 25/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_RATIONAL_CONVERSION_H
#define ROUGHPY_GENERICS_INTERNAL_RATIONAL_CONVERSION_H


#include <memory>

#include "roughpy/core/macros.h"

#include "roughpy/generics/type_ptr.h"
#include "roughpy/generics/conversion_trait.h"

namespace rpy {
namespace generics {

class MPRationalConversionFromFactory {
public:
    virtual ~MPRationalConversionFromFactory() = default;
    virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;

    static const MPRationalConversionFromFactory* get_factory(const Type& type) noexcept;
};


class MPRationalConversionToFactory {
public:
    virtual ~MPRationalConversionToFactory() = default;

    virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;

    static const MPRationalConversionToFactory* get_factory(const Type& type) noexcept;
};

} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_RATIONAL_CONVERSION_H
