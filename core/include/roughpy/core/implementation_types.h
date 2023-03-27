#ifndef ROUGHPY_CORE_IMPLEMENTATION_TYPES_H_
#define ROUGHPY_CORE_IMPLEMENTATION_TYPES_H_

#include <cstdint>
#include <utility>


#include <boost/optional.hpp>

namespace rpy {

using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;

using let_t = uint16_t;
using dimn_t = std::size_t;
using idimn_t = std::ptrdiff_t;
using deg_t = int;
using key_type = std::size_t;
using param_t = double;
using scalar_t = double;

using dyadic_multiplier_t = int;
using dyadic_depth_t = int;
using resolution_t = dyadic_depth_t;


using boost::optional;

using std::pair;


}

#endif // ROUGHPY_CORE_IMPLEMENTATION_TYPES_H_
