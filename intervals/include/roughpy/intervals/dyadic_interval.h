#ifndef ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_
#define ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_

#include "dyadic.h"
#include "interval.h"

#include <iosfwd>
#include <vector>

namespace rpy {
namespace intervals {

class ROUGHPY_INTERVALS_EXPORT DyadicInterval : public Dyadic, public Interval {
    using Dyadic::m_multiplier;
    using Dyadic::m_power;

protected:
    using Interval::m_interval_type;

public:
    using typename Dyadic::multiplier_t;
    using typename Dyadic::power_t;

    using Dyadic::Dyadic;

    DyadicInterval() = default;

    explicit DyadicInterval(IntervalType itype) : Dyadic(), Interval(itype) {
        assert(itype == IntervalType::Clopen || itype == IntervalType::Opencl);
    }
    explicit DyadicInterval(Dyadic dyadic) : Dyadic(dyadic), Interval() {}
    DyadicInterval(multiplier_t k, power_t n, IntervalType itype) : Dyadic(k, n), Interval(itype) {
        assert(itype == IntervalType::Clopen || itype == IntervalType::Opencl);
    }
    DyadicInterval(Dyadic dyadic, power_t resolution, IntervalType itype = IntervalType::Clopen);
    DyadicInterval(param_t val, power_t resolution, IntervalType itype = IntervalType::Clopen);

    multiplier_t unit() const noexcept {
        return (m_interval_type == IntervalType::Clopen) ? 1 : -1;
    }

    using Dyadic::operator++;
    using Dyadic::operator--;
    using Dyadic::multiplier;
    using Dyadic::power;

    Dyadic dincluded_end() const;
    Dyadic dexcluded_end() const;
    Dyadic dinf() const;
    Dyadic dsup() const;

    param_t inf() const override;
    param_t sup() const override;
    param_t included_end() const override;
    param_t excluded_end() const override;

    DyadicInterval shrink_to_contained_end(power_t arg = 1) const;
    DyadicInterval shrink_to_omitted_end() const;
    DyadicInterval &shrink_interval_right();
    DyadicInterval &shrink_interval_left(power_t arg = 1);
    DyadicInterval &expand_interval(power_t arg = 1);

    bool contains(const DyadicInterval &other) const;
    bool aligned() const;

    DyadicInterval &flip_interval();

    DyadicInterval shift_forward(multiplier_t arg = 1) const;
    DyadicInterval shift_back(multiplier_t arg = 1) const;

    DyadicInterval &advance() noexcept;
};


ROUGHPY_INTERVALS_EXPORT
std::ostream& operator<<(std::ostream& os, const DyadicInterval& di);


ROUGHPY_INTERVALS_EXPORT
std::vector<DyadicInterval> to_dyadic_intervals(const Interval& interval, typename Dyadic::power_t tol, IntervalType itype=IntervalType::Clopen);


}// namespace intervals
}// namespace rpy

#endif// ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_
