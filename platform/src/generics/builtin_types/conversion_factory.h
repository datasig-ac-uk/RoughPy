//
// Created by sammorley on 17/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_CONVERSION_FACTORY_H
#define ROUGHPY_GENERICS_INTERNAL_CONVERSION_FACTORY_H


#include <typeinfo>

#include <boost/container/flat_map.hpp>

#include "roughpy/core/hash.h"

#include "generics/conversion_trait.h"
#include "generics/type.h"

#include "builtin_type_ids.h"


namespace rpy {
namespace generics {

class RPY_LOCAL ConversionFactory {

public:

    virtual ~ConversionFactory();

    RPY_NO_DISCARD
    virtual std::unique_ptr<const ConversionTrait> make(TypePtr from_type, TypePtr to_type) const = 0;

};


template <typename From, typename To>
class RPY_LOCAL ConversionFactoryImpl : public ConversionFactory
{
    ConversionFactoryImpl() = default;
public:

    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait> make(TypePtr from_type, TypePtr to_type) const override;

    RPY_NO_DISCARD
    const ConversionFactory* get() const noexcept
    {
        static const ConversionFactoryImpl object;
        return &object;
    }
};



template <typename From, typename To>
std::unique_ptr<const ConversionTrait>
ConversionFactoryImpl<From, To>::make(TypePtr from_type, TypePtr to_type) const
{
    RPY_DBG_ASSERT_EQ(from_type->type_info(), typeid(From));
    RPY_DBG_ASSERT_EQ(to_type->type_info(), typeid(Type));

    return std::make_unique<const ConversionTraitImpl<From, To>>(
        std::move(from_type), std::move(to_type)
    );
}


namespace dtl {


template <typename... Ts>
struct TypeList
{
    static constexpr size_t size = sizeof...(Ts);
};


template <typename BaseType, typename Map, typename T, typename... Ts>
inline void construct_from_table(Map& map, TypeList<T, Ts...> list)
{
    Hash<string_view> hasher;
    map.emplace(
        hasher(type_id_of<T>),
            ConversionFactoryImpl<BaseType, T>::get()
    );
    if constexpr (sizeof...(Ts) > 0) {
        construct_from_table(map, TypeList<Ts...>{});
    }
}

template <typename ToType, typename Map, typename T, typename... Ts>
void construct_to_table(Map& map, TypeList<T, Ts...> list)
{
    Hash<string_view> hasher;
    map.emplace(
        hasher(type_id_of<T>),
        ConversionFactoryImpl<T, ToType>::get()
        );
    if constexpr (sizeof...(Ts) > 0) {
        consturct_to_table(map, TypeList<Ts...>{});
    }
}


}


template <typename BaseType>
boost::container::flat_map<hash_t, const ConversionFactory*>
make_conversion_to_table()
{
    using type_list = dtl::TypeList<
        uint8_t, int8_t,
        uint16_t, int16_t,
        uint32_t, int32_t,
        uint64_t, int64_t,
        float, double>;

    boost::container::flat_map<hash_t, const ConversionFactory*> map;
    map.reserve(type_list::size);

    dtl::construct_to_table<BaseType>(map, type_list{});

    return map;
}


template <typename ToType>
boost::container::flat_map<hash_t, const ConversionFactory*>
make_conversion_from_table()
{
    using type_list = dtl::TypeList<
        uint8_t, int8_t,
        uint16_t, int16_t,
        uint32_t, int32_t,
        uint64_t, int64_t,
        float, double>;

    boost::container::flat_map<hash_t, const ConversionFactory*> map;
    map.reserve(type_list::size);

    dtl::construct_from_table<ToType>(map, type_list{});

    return map;
}

} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_CONVERSION_FACTORY_H