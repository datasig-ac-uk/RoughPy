#ifndef ROUGHPY_INTERVALS_SEGMENTATION_H_
#define ROUGHPY_INTERVALS_SEGMENTATION_H_

#include "interval.h"
#include "real_interval.h"

#include <functional>
#include <vector>

namespace rpy { namespace intervals {


using predicate_t = std::function<bool(const Interval&)>;

ROUGHPY_INTERVALS_EXPORT
std::vector<RealInterval>
segment(const Interval& interval,
        predicate_t predicate,
        dyadic_depth_t max_depth
        );



}}


#endif // ROUGHPY_INTERVALS_SEGMENTATION_H_
