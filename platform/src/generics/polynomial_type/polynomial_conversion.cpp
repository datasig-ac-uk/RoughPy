//
// Created by sammorley on 02/12/24.
//

#include "polynomial_conversion.h"

#include "generics/builtin_types/builtin_type_ids.h"
#include "generics/builtin_types/conversion_helpers.h"

#include "generics/multiprecision_types/multiprecision_type_ids.h"
#include "generics/multiprecision_types/conversion_helpers.h"


#include "polynomial.h"
#include "conversion_helpers.h"
#include "polynomial_type_id.h"

using namespace rpy;
using namespace rpy::generics;


boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory
>> conv::make_poly_conversion_to_table()
{
    using type_list = BuiltinTypesList::Append<MPFloat, MPRational, MPInt>;
    boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory>> table;
    table.reserve(type_list::size);

    conv::build_conversion_to_table<Polynomial>(table, type_list{});

    return table;
}

boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory
>> conv::make_poly_conversion_from_table()
{
    using type_list = BuiltinTypesList::Append<MPFloat, MPRational, MPInt>;
    boost::container::flat_map<hash_t, std::unique_ptr<const conv::ConversionFactory>> table;
    table.reserve(type_list::size);

    conv::build_conversion_from_table<Polynomial>(table, type_list{});

    return table;
}
