//
// Created by sammorley on 25/11/24.
//

#include "rational_conversion.h"

#include <limits>

#include <gmp.h>
#include <mpfr.h>
#include <boost/container/flat_map.hpp>

#include "generics/builtin_types/builtin_type_ids.h"

#include "rational_type.h"
#include "multiprecision_type_ids.h"


using namespace rpy;
using namespace rpy::generics;

boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory
>> conv::make_mprational_conversion_to_table()
{
    using type_list = BuiltinTypesList::Append<MPInt>;
    boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory>> table;
    table.reserve(type_list::size);

    conv::build_conversion_to_table<MPRational>(table, type_list{});

    return table;
}

boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory
>> conv::make_mprational_conversion_from_table()
{
    using type_list = BuiltinTypesList::Append<MPInt>;
    boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory>> table;
    table.reserve(type_list::size);

    conv::build_conversion_from_table<MPRational>(table, type_list{});

    return table;
}
