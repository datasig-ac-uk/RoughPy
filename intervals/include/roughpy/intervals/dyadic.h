#ifndef ROUGHPY_INTERVALS_DYADIC_H_
#define ROUGHPY_INTERVALS_DYADIC_H_

#include "roughpy_intervals_export.h"

#include <cassert>
#include <cmath>
#include <iosfwd>
#include <limits>

namespace rpy { namespace intervals {

using param_t = double;

class ROUGHPY_INTERVALS_EXPORT Dyadic {
public:
    using multiplier_t = int;
    using power_t = int;

protected:
    multiplier_t m_multiplier = 0;
    power_t m_power = 0;

public:
    static constexpr multiplier_t mod(multiplier_t a, multiplier_t b) {
        multiplier_t r = a % b;
        return (r < 0) ? (r + abs(b)) : r;
    }
    static constexpr multiplier_t int_two_to_int_power(power_t exponent) {
        assert(exponent >= 0);
        return multiplier_t(1) << exponent;
    }
    static constexpr multiplier_t shift(multiplier_t k, power_t n) {
        return k * int_two_to_int_power(n);
    }

    Dyadic() = default;

    explicit Dyadic(multiplier_t k, power_t n=0) : m_multiplier(k), m_power(n)
    {}


    multiplier_t multiplier() const noexcept { return m_multiplier; }
    power_t power() const noexcept { return m_power; }

    explicit operator param_t() const noexcept;


    Dyadic& move_forward(multiplier_t arg);
    Dyadic& operator++();
    const Dyadic operator++(int);
    Dyadic& operator--();
    const Dyadic operator--(int);

    bool rebase(power_t resolution=std::numeric_limits<power_t>::lowest());




};

ROUGHPY_INTERVALS_EXPORT
bool operator<(const Dyadic& lhs, const Dyadic& rhs);

ROUGHPY_INTERVALS_EXPORT
bool operator<=(const Dyadic& lhs, const Dyadic& rhs);

ROUGHPY_INTERVALS_EXPORT
bool operator>(const Dyadic& lhs, const Dyadic& rhs);

ROUGHPY_INTERVALS_EXPORT
bool operator>=(const Dyadic& lhs, const Dyadic& rhs);

ROUGHPY_INTERVALS_EXPORT
std::ostream& operator<<(std::ostream& os, const Dyadic& arg);

ROUGHPY_INTERVALS_EXPORT
bool dyadic_equals(const Dyadic& lhs, const Dyadic& rhs);

ROUGHPY_INTERVALS_EXPORT
bool rational_equals(const Dyadic& lhs, const Dyadic& rhs);


}}


#endif // ROUGHPY_INTERVALS_DYADIC_H_
