//
// Created by sammorley on 17/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_CONVERSION_FACTORY_H
#define ROUGHPY_GENERICS_INTERNAL_CONVERSION_FACTORY_H


#include <boost/container/flat_map.hpp>

#include "roughpy/core/hash.h"

#include "roughpy/generics/conversion_trait.h"
#include "roughpy/generics/type.h"

#include "generics/conversion_impl.h"

#include "builtin_type_ids.h"
#include "conversion_helpers.h"

namespace rpy {
namespace generics::conv{


template <typename BaseType>
boost::container::flat_map<hash_t, std::unique_ptr<const ConversionFactory>>
make_conversion_to_table()
{
    using type_list = BuiltinTypesList;
    boost::container::flat_map<hash_t, std::unique_ptr<const ConversionFactory>> map;
    map.reserve(type_list::size);

    build_conversion_to_table<BaseType>(map, type_list{});

    return map;
}

template <typename ToType>
boost::container::flat_map<hash_t, std::unique_ptr<const ConversionFactory>>
make_conversion_from_table()
{
    using type_list = BuiltinTypesList;

    boost::container::flat_map<hash_t, std::unique_ptr<const ConversionFactory>> map;
    map.reserve(type_list::size);

    build_conversion_from_table<ToType>(map, type_list{});

    return map;
}

}// namespace generics
}// namespace rpy

#endif// ROUGHPY_GENERICS_INTERNAL_CONVERSION_FACTORY_H