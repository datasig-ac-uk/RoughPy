//
// Created by user on 02/03/23.
//


#include "interval.h"
#include <stdexcept>
#include <ostream>

using namespace rpy;
using namespace rpy::intervals;

param_t Interval::included_end() const {
    if (m_interval_type == IntervalType::Clopen) {
        return inf();
    }
    if (m_interval_type == IntervalType::Opencl) {
        return sup();
    }
    throw std::runtime_error("included_end is not valid for intervals that are not half open");
}
param_t Interval::excluded_end() const {
    if (m_interval_type == IntervalType::Clopen) {
        return sup();
    }
    if (m_interval_type == IntervalType::Opencl) {
        return inf();
    }
    throw std::runtime_error("excluded_end is not valid for intervals that are not half open");
}
bool Interval::contains(param_t arg) const noexcept {
    if (m_interval_type == IntervalType::Clopen) {
        return inf() <= arg && arg < sup();
    }
    if (m_interval_type == IntervalType::Opencl) {
        return inf() < arg && arg <= sup();
    }


    return false;
}
bool Interval::is_associated(const Interval &arg) const noexcept {
    return contains(arg.included_end());
}
bool Interval::contains(const Interval &arg) const noexcept {
    return contains(arg.inf()) && contains(arg.sup());
}
bool Interval::intersects_with(const Interval &arg) const noexcept {
    auto lhs_inf = inf();
    auto lhs_sup = sup();
    auto rhs_inf = arg.inf();
    auto rhs_sup = arg.sup();

    if ((lhs_inf <= rhs_inf && lhs_sup > rhs_inf) || (rhs_inf <= lhs_inf && rhs_sup > lhs_inf)) {
        // [l--[r---l)--r) || [r--[l--r)--l)
        return true;
    }
    if (rhs_inf == lhs_sup) {
        // (l--l][r--r)
        return m_interval_type == IntervalType::Opencl && arg.m_interval_type == IntervalType::Clopen;
    }
    if (lhs_inf == rhs_sup) {
        // (r--r][l---l)
        return m_interval_type == IntervalType::Clopen && arg.m_interval_type == IntervalType::Opencl;
    }
    return false;
}
bool Interval::operator==(const Interval &other) const {
    return m_interval_type == other.m_interval_type && inf() == other.inf() && sup() == other.sup();
}
bool Interval::operator!=(const Interval &other) const {
    return !operator==(other);
}

std::ostream &rpy::intervals::operator<<(std::ostream &os, const Interval &interval) {
    if (interval.type() == IntervalType::Clopen) {
        os << '[';
    } else {
        os << '(';
    }

    os << interval.inf() << ", " << interval.sup();

    if (interval.type() == IntervalType::Clopen) {
        os << ')';
    } else {
        os << ']';
    }

    return os;
}
