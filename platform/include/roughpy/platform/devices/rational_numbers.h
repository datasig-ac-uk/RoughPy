//
// Created by sam on 12/11/23.
//

#ifndef ROUGHPY_DEVICE_RATIONAL_NUMBERS_H
#define ROUGHPY_DEVICE_RATIONAL_NUMBERS_H

#include <boost/multiprecision/gmp.hpp>
#include <libalgebra_lite/coefficients.h>
#include <libalgebra_lite/polynomial.h>


namespace rpy {
namespace devices {


using rational_scalar_type = boost::multiprecision::mpq_rational;
using rational_field = lal::coefficient_field<rpy::devices::rational_scalar_type>;

}
}

namespace lal {

template <>
struct coefficient_trait<rpy::devices::rational_scalar_type> {
    using coefficient_ring = rpy::devices::rational_field;
    using scalar_type = rpy::devices::rational_scalar_type;
    using rational_type = rpy::devices::rational_scalar_type;
};

template <>
struct coefficient_ring<lal::polynomial<rpy::devices::rational_field>, rpy::devices::rational_scalar_type> {
    using scalar_type = lal::polynomial<rpy::devices::rational_field>;
    using rational_type = rpy::devices::rational_scalar_type;

    static const scalar_type &zero() noexcept {
        static const scalar_type zero;
        return zero;
    }

    static const scalar_type &one() noexcept {
        static const scalar_type one(1);
        return one;
    }

    static const scalar_type &mone() noexcept {
        static const scalar_type mone(-1);
        return mone;
    }

    static inline bool is_invertible(const scalar_type &arg) {
        return arg.size() == 1
               && arg.degree() == 0
               && rpy::devices::rational_field::is_invertible(arg.begin()->value());
    }

    static inline const rational_type &as_rational(const scalar_type &arg)
    noexcept {
        return rpy::devices::rational_field::as_rational(arg.begin()->value());
    }

};


extern template class polynomial<rpy::devices::rational_field>;

}


namespace rpy {
namespace devices {


using rational_poly_scalar = lal::polynomial<rational_field>;
}
}


#endif //ROUGHPY_DEVICE_RATIONAL_NUMBERS_H
