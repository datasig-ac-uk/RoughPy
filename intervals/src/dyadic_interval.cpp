//
// Created by user on 02/03/23.
//

#include "dyadic_interval.h"

#include <algorithm>
#include <list>


#include "real_interval.h"

using namespace rpy::intervals;

DyadicInterval::DyadicInterval(Dyadic dyadic,
                               Dyadic::power_t resolution,
                               IntervalType itype)
    : Dyadic(dyadic), Interval(itype) {
    if (!rebase(resolution)) {
        multiplier_t k1 = m_multiplier;
        const multiplier_t one = unit();
        multiplier_t pow = int_two_to_int_power(m_power - resolution);
        m_multiplier = one * (k1 * one - mod(k1 * one, pow));
        bool is_int = rebase(resolution);
        assert(is_int);
    }
}
DyadicInterval::DyadicInterval(param_t val,
                               Dyadic::power_t resolution,
                               IntervalType itype)
    : Dyadic(0, 0), Interval(itype)
{
    auto rescaled = ldexp(val, resolution);
    if (m_interval_type == IntervalType::Opencl) {
        m_multiplier = ceil(rescaled);
    } else {
        m_multiplier = floor(rescaled);
    }
    m_power = resolution;
}
Dyadic DyadicInterval::dincluded_end() const {
    return static_cast<const Dyadic&>(*this);
}
Dyadic DyadicInterval::dexcluded_end() const {
    return Dyadic(m_multiplier + unit(), m_power);
}
Dyadic DyadicInterval::dinf() const {
    return m_interval_type == IntervalType::Clopen
        ? dincluded_end()
        : dexcluded_end();
}
Dyadic DyadicInterval::dsup() const {
    return m_interval_type == IntervalType::Opencl
        ? dexcluded_end()
        : dincluded_end();
}
param_t DyadicInterval::inf() const {
    return static_cast<param_t>(dinf());
}
param_t DyadicInterval::sup() const {
    return static_cast<param_t>(dsup());
}
param_t DyadicInterval::included_end() const {
    return static_cast<param_t>(dincluded_end());
}
param_t DyadicInterval::excluded_end() const {
    return static_cast<param_t>(dexcluded_end());
}

DyadicInterval DyadicInterval::shrink_to_contained_end(Dyadic::power_t arg) const {
    return {static_cast<const Dyadic &>(*this), arg + m_power, m_interval_type};
}
DyadicInterval DyadicInterval::shrink_to_omitted_end() const {
    return shrink_to_contained_end().flip_interval();
}
DyadicInterval &DyadicInterval::shrink_interval_right() {
    if (m_interval_type == IntervalType::Opencl) {
        *this = shrink_to_contained_end();
    } else {
        *this = shrink_to_omitted_end();
    }
    return *this;
}
DyadicInterval &DyadicInterval::shrink_interval_left(Dyadic::power_t arg) {
    assert(arg >= 0);
    for (; arg > 0; --arg) {
        if (m_interval_type == IntervalType::Clopen) {
            *this = shrink_to_contained_end();
        } else {
            *this = shrink_to_omitted_end();
        }
    }
    return *this;
}
DyadicInterval &DyadicInterval::expand_interval(Dyadic::power_t arg) {
    *this = DyadicInterval{dincluded_end(), m_power - arg};
    return *this;
}
bool DyadicInterval::contains(const DyadicInterval &other) const {
    if (other.m_interval_type != m_interval_type) {
        return false;
    }
    if (other.m_power >= m_power) {
        multiplier_t one(unit());
        multiplier_t pow = int_two_to_int_power(other.m_power - m_power);
        multiplier_t shifted = shift(m_multiplier, (other.m_power - m_power));
        multiplier_t aligned = one * (other.m_multiplier * one - mod(other.m_multiplier * one, pow));

        return shifted == aligned;
    }
    return false;
}
bool DyadicInterval::aligned() const {
    DyadicInterval parent{static_cast<const Dyadic &>(*this), m_power - 1, m_interval_type};
    return operator==(DyadicInterval{static_cast<const Dyadic &>(parent), m_power});
}
DyadicInterval &DyadicInterval::flip_interval() {
    if ((m_multiplier % 2) == 0) {
        m_multiplier += unit();
    } else {
        m_multiplier -= unit();
    }
    return *this;
}
DyadicInterval DyadicInterval::shift_forward(Dyadic::multiplier_t arg) const {
    DyadicInterval tmp(*this);
    tmp.m_multiplier -= unit() * arg;
    return tmp;
}
DyadicInterval DyadicInterval::shift_back(Dyadic::multiplier_t arg) const {
    DyadicInterval tmp(*this);
    tmp.m_multiplier += unit() * arg;
    return tmp;
}
DyadicInterval &DyadicInterval::advance() noexcept {
    m_multiplier += unit();
    return *this;
}

std::ostream &rpy::intervals::operator<<(std::ostream &os, const DyadicInterval &di) {
    return os << static_cast<const Interval &>(di);
}

std::vector<DyadicInterval> rpy::intervals::to_dyadic_intervals(const Interval &interval, Dyadic::power_t tol, IntervalType itype) {
    using iterator = std::list<DyadicInterval>::iterator;
    std::list<DyadicInterval> intervals;

    auto store_move = [&](DyadicInterval &b) {
        intervals.push_back(b.shrink_to_omitted_end());
        b.advance();
    };

    auto store_ = [&](iterator &p, DyadicInterval &e) -> iterator {
        return intervals.insert(p, e.shrink_to_contained_end());
    };

    RealInterval real{interval, itype};

    DyadicInterval begin{real.included_end(), tol, itype};
    DyadicInterval end{real.excluded_end(), tol, itype};

    while (!begin.contains(end)) {
        auto next{begin};
        next.expand_interval();
        if (!begin.aligned()) {
            store_move(next);
        }
        begin = std::move(next);
    }

    auto p = intervals.end();
    for (auto next{end}; begin.contains(next.expand_interval());) {
        if (!end.aligned()) {
            p = store_(p, next);
        }
        end = next;
    }

    if (itype == IntervalType::Opencl) {
        intervals.reverse();
    }

    return {intervals.begin(), intervals.end()};
}
