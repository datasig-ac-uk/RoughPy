//
// Created by sam on 3/26/24.
//

#ifndef ARBITRARY_PRECISION_RATIONAL_H
#define ARBITRARY_PRECISION_RATIONAL_H

#include <boost/multiprecision/gmp.hpp>

#define RPY_USING_GMP 1

namespace rpy { namespace scalars {



using ArbitraryPrecisionRational = boost::multiprecision::mpq_rational;


// template <>
// struct type_id_of_impl<ArbitraryPrecisionRational>
// {
//     static const string& get_id() noexcept;
// };

}

namespace devices {

namespace dtl {
template <>
struct type_code_of_impl<scalars::ArbitraryPrecisionRational>
{
    static constexpr TypeCode value = TypeCode::ArbitraryPrecisionRational;
};

}
}

}

#endif //ARBITRARY_PRECISION_RATIONAL_H
