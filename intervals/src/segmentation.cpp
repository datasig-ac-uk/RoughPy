//
// Created by user on 02/03/23.
//

#include "segmentation.h"
#include "dyadic_interval.h"

#include "dyadic_searcher.h"

/*
 * We have to re-work the segmentation library pysegments to work with our interval types, rather
 * that those defined in libRDE
 */


using namespace rpy::intervals;

std::vector<RealInterval>
rpy::intervals::segment(const Interval &interval, predicate_t predicate, dyadic_depth_t max_depth) {
    DyadicSearcher search(std::move(predicate), max_depth);
    return search(interval);
}
