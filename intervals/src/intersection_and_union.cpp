//
// Created by sammorley on 03/12/24.
//


#include "interval.h"

#include <cmath>

#include "real_interval.h"

using namespace rpy;
using namespace rpy::intervals;


RealInterval intervals::intersection(const Interval& lhs,
                                     const Interval& rhs) noexcept
{
    auto linf = lhs.inf();
    auto rinf = rhs.inf();
    auto lsup = lhs.sup();
    auto rsup = rhs.sup();

    if (linf >= rsup || lsup <= rinf) { return RealInterval(); }

    auto inf = std::max(linf, rinf);
    auto sup = std::min(lsup, rsup);

    return RealInterval(inf, sup);
}


RealInterval
intervals::interval_union(const Interval& lhs, const Interval& rhs) noexcept
{
    return RealInterval(std::min(lhs.inf(), rhs.inf()),
                        std::max(lhs.sup(), rhs.sup()));
}