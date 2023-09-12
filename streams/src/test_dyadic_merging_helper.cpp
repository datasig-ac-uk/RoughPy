//
// Created by user on 12/09/23.
//



#include <gtest/gtest.h>

#include "dyadic_merging_helper.h"

using namespace rpy;
using namespace rpy::streams;
using namespace rpy::intervals;

TEST(DyadicMergingHelper, UnitIntervalPower2) {
    DyadicMergingHelper helper(2);

    RealInterval unit(0, 1);

    auto range = helper.insert(unit);

    EXPECT_EQ(range.lower, 0);
    EXPECT_EQ(range.upper, 4);

    auto dyadics = helper.to_dyadics();

    std::vector<DyadicInterval> expected_di;
    expected_di.reserve(4);
    for (int i=0; i<4; ++i) {
        expected_di.emplace_back(i, 2);
    }

    EXPECT_EQ(expected_di, dyadics);
}
