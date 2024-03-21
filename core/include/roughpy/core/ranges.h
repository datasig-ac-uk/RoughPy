//
// Created by sam on 21/03/24.
//

#ifndef ROUGHPY_CORE_RANGES_H
#define ROUGHPY_CORE_RANGES_H

#include "macros.h"
#include "types.h"

#include <boost/range.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/adaptor/copied.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/type_erased.hpp>
#include <boost/range/combine.hpp>

namespace rpy {

namespace ranges {



template <typename... Ranges>
auto zip(Ranges&&... ranges) -> decltype(boost::combine(std::forward<Ranges>(ranges)...))
{
    return boost::combine(std::forward<Ranges>(ranges)...);
}

using boost::adaptors::adjacent_filtered;
using boost::adaptors::copied;
using boost::adaptors::indexed;
using boost::adaptors::map_keys;
using boost::adaptors::map_values;
using boost::adaptors::replaced;
using boost::adaptors::replaced_if;
using boost::adaptors::reversed;
using boost::adaptors::sliced;
using boost::adaptors::type_erased;
using boost::adaptors::tokenized;
using boost::adaptors::transformed;
using boost::adaptors::uniqued;


using boost::iterator_range;


}}



#endif// ROUGHPY_CORE_RANGES_H
