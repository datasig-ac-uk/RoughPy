//
// Created by sam on 12/05/23.
//

#include <gtest/gtest.h>

#include <roughpy/intervals/real_interval.h>

#include <sstream>

using namespace rpy;
using namespace rpy::intervals;

TEST(RealInterval, RealIntervalSerialization)
{

    RealInterval interval(2.0, 5.0);
    std::stringstream ss;
    {
        archives::JSONOutputArchive oarc(ss);
        oarc(interval);
    }

    RealInterval in_terval;
    {
        archives::JSONInputArchive iarc(ss);
        iarc(in_terval);
    }

    EXPECT_EQ(in_terval, interval);
}
