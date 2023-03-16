//
// Created by user on 02/03/23.
//


#include "interval.h"
#include <stdexcept>
#include <ostream>

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
    return false;
}
bool Interval::contains(const Interval &arg) const noexcept {
    return contains(arg.inf()) && contains(arg.sup());
}
bool Interval::intersects_with(const Interval &arg) const noexcept {
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
