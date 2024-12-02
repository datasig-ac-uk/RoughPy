//
// Created by sam on 28/11/24.
//

#include "integer_conversion.h"

#include <gmp.h>
#include <mpfr.h>

#include "generics/builtin_types/builtin_type_ids.h"

#include "integer_type.h"
#include "multiprecision_type_ids.h"


using namespace rpy;
using namespace rpy::generics;

boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory
>> conv::make_mpint_conversion_to_table()
{
    using type_list = BuiltinTypesList;
    boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory>> table;
    table.reserve(type_list::size);

    conv::build_conversion_to_table<MPInt>(table, type_list{});

    return table;
}

boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory
>> conv::make_mpint_conversion_from_table()
{
    using type_list = BuiltinTypesList;
    boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory>> table;
    table.reserve(type_list::size);

    conv::build_conversion_from_table<MPInt>(table, type_list{});

    return table;
}
