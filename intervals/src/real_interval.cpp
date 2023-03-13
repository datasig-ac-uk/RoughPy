//
// Created by user on 02/03/23.
//

#include "real_interval.h"

bool rpy::intervals::RealInterval::contains(const Interval &arg) const noexcept {
    return Interval::contains(arg);
}
