//
// Created by sam on 24/06/24.
//

#ifndef ARBITRARY_PRECISION_RATIONAL_H
#define ARBITRARY_PRECISION_RATIONAL_H


#include <roughpy/core/types.h>
#include <roughpy/devices/core.h>

#include <boost/multiprecision/gmp.hpp>

#define RPY_USING_GMP 1


namespace rpy {
namespace scalars {
namespace implementations {

using ArbitraryPrecisionRational = boost::multiprecision::mpq_rational;


}
}

namespace devices {


namespace dtl {
template <>
struct type_code_of_impl<scalars::implementations::ArbitraryPrecisionRational> {
    static constexpr TypeCode value = TypeCode::ArbitraryPrecisionRational;
};
template <>
struct type_id_of_impl<scalars::implementations::ArbitraryPrecisionRational>
{
    static constexpr string_view value = "Rational";
};

}

}
}


#endif //ARBITRARY_PRECISION_RATIONAL_H
