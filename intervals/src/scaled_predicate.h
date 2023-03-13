//
// Created by user on 02/03/23.
//

#ifndef ROUGHPY_INTERVALS_SRC_SCALED_PREDICATE_H
#define ROUGHPY_INTERVALS_SRC_SCALED_PREDICATE_H

#include "segmentation.h"
#include "dyadic_interval.h"

namespace rpy {
namespace intervals {

class ScaledPredicate {
    predicate_t& m_func;
    param_t m_shift;
    param_t m_scale;

public:

    ScaledPredicate(predicate_t& predicate,
                    param_t shift,
                    param_t scale)
        : m_func(predicate),
          m_shift(shift),
          m_scale(scale)
    {}

    RealInterval unscale(const Interval& interval) const;

    bool operator()(const Interval& interval) const;


};

}// namespace intervals
}// namespace rpy

#endif//ROUGHPY_INTERVALS_SRC_SCALED_PREDICATE_H
