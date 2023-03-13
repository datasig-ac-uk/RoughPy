//
// Created by user on 03/03/23.
//

#ifndef ROUGHPY_INTERVALS_SRC_DYADIC_SEARCHER_H
#define ROUGHPY_INTERVALS_SRC_DYADIC_SEARCHER_H

#include "segmentation.h"

#include <cassert>
#include <deque>
#include <map>
#include <utility>

#include "dyadic.h"

#include "scaled_predicate.h"

namespace rpy {
namespace intervals {

struct DyadicRealStrictLess {
    bool operator()(const Dyadic& lhs, const Dyadic& rhs) const noexcept {
        auto max = std::max(lhs.power(), rhs.power());
        return (lhs.multiplier() << (max - lhs.power())) < (rhs.multiplier() << (max - rhs.power()));
    }
};

struct DyadicRealStrictGreater {
    bool operator()(const Dyadic& lhs, const Dyadic& rhs) const noexcept {
        auto max = std::max(lhs.power(), rhs.power());
        return (lhs.multiplier() << (max - lhs.power())) > (rhs.multiplier() << (max - rhs.power()));
    }
};

class DyadicSearcher {
    predicate_t m_predicate;
    std::map<Dyadic, Dyadic, DyadicRealStrictGreater> m_seen;
    dyadic_depth_t m_max_depth;

protected:

    void expand_left(ScaledPredicate& predicate, std::deque<DyadicInterval>& current) const;
    void expand_right(ScaledPredicate& predicate, std::deque<DyadicInterval>& current) const;
    void expand(ScaledPredicate& predicate, DyadicInterval found_interval);

public:

    DyadicSearcher(predicate_t&& predicate, dyadic_depth_t max_depth)
        : m_predicate(std::move(predicate)), m_max_depth(max_depth)
    {}

private:

    ScaledPredicate rescale_to_unit_interval(const Interval& original);
    void get_next_dyadic(DyadicInterval& current) const;
    std::vector<RealInterval> find_in_unit_interval(ScaledPredicate& predicate);

public:

    std::vector<RealInterval> operator()(const Interval& original);


};

}// namespace intervals
}// namespace rpy

#endif//ROUGHPY_INTERVALS_SRC_DYADIC_SEARCHER_H
