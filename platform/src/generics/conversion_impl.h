//
// Created by sammorley on 02/12/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_CONVERSION_IMPL_H
#define ROUGHPY_GENERICS_INTERNAL_CONVERSION_IMPL_H


#include <memory>

#include "roughpy/core/hash.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/meta.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/generics/conversion_trait.h"
#include "roughpy/generics/type_ptr.h"
#include "roughpy/generics/type.h"


namespace rpy::generics::conv {






/**
 * @brief This class is responsible for creating type conversion objects.
 *
 * The ConversionFactory class provides a factory mechanism to generate instances
 * of type conversion objects. It manages the creation process and simplifies
 * the instantiation of converters for various type transformations by delegating
 * the responsibility of object creation to the factory class.
 */
class ConversionFactory
{
public:
    virtual ~ConversionFactory();

    RPY_NO_DISCARD
    virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;
};

template <typename From, typename To>
class ConversionFactoryImpl : public ConversionFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename From, typename To>
std::unique_ptr<const ConversionTrait> ConversionFactoryImpl<From, To>::make(
    TypePtr from_type,
    TypePtr to_type) const
{
    return std::make_unique<const ConversionTraitImpl<From, To>>(
            std::move(from_type),
            std::move(to_type)
    );
}


template <typename From, typename Map, typename To, typename... Ts>
void build_conversion_from_table(Map& map, meta::TypeList<To, Ts...> list)
{
    using helper = ConversionHelper<From, To>;
    ignore_unused(list);
    if constexpr (!is_same_v<From, To> && helper::is_possible) {
        Hash<string_view> hasher;
        map.emplace(
            hasher(type_id_of<To>),
            std::make_unique<ConversionFactoryImpl<From, To>>()
        );
    }
    if constexpr (sizeof...(Ts) > 0) {
        build_conversion_from_table<From>(map, meta::TypeList<Ts...>{});
    }
}



template <typename To, typename Map, typename From, typename... Ts>
void build_conversion_to_table(Map& map, meta::TypeList<From, Ts...> list)
{
    ignore_unused(list);
    using helper = ConversionHelper<From, To>;
    if constexpr (!is_same_v<From, To> && helper::is_possible) {
        Hash<string_view> hasher;
        map.emplace(
            hasher(type_id_of<From>),
            std::make_unique<ConversionFactoryImpl<From, To>>()
        );
    }
    if constexpr (sizeof...(Ts) > 0) {
        build_conversion_from_table<To>(map, meta::TypeList<Ts...>{});
    }
}






} // namespace rpy::generics::conv


#endif //ROUGHPY_GENERICS_INTERNAL_CONVERSION_IMPL_H
