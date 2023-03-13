//
// Created by user on 02/03/23.
//

#include "scaled_predicate.h"



using namespace rpy::intervals;

RealInterval ScaledPredicate::unscale(const Interval &interval) const {
    return {
        interval.inf()*m_scale + m_shift,
        interval.sup()*m_scale + m_shift,
        interval.type()
    };

}
bool ScaledPredicate::operator()(const Interval &interval) const {
    return m_func(unscale(interval));
}
