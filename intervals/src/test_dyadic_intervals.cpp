// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 16/03/23.
//

#include <gtest/gtest.h>

#include <roughpy/intervals/dyadic_interval.h>
#include <roughpy/intervals/real_interval.h>

#include <sstream>

using namespace rpy;
using namespace rpy::intervals;

TEST(DyadicIntervals, test_dincluded_end_Clopen)
{
    DyadicInterval di;
    Dyadic expected{0, 0};
    ASSERT_TRUE(dyadic_equals(di.dincluded_end(), expected));
}

TEST(DyadicIntervals, test_dexcluded_end_Clopen)
{
    DyadicInterval di;
    Dyadic expected{1, 0};
    ASSERT_TRUE(dyadic_equals(di.dexcluded_end(), expected));
}

TEST(DyadicIntervals, test_dincluded_end_Opencl)
{
    DyadicInterval di{IntervalType::Opencl};
    Dyadic expected{0, 0};
    ASSERT_TRUE(dyadic_equals(di.dincluded_end(), expected));
}

TEST(DyadicIntervals, test_dexcluded_end_Opencl)
{
    DyadicInterval di{IntervalType::Opencl};
    Dyadic expected{-1, 0};
    ASSERT_TRUE(dyadic_equals(di.dexcluded_end(), expected));
}

TEST(DyadicIntervals, test_included_end_Clopen)
{
    DyadicInterval di;

    ASSERT_EQ(di.included_end(), 0.0);
}

TEST(DyadicIntervals, test_excluded_end_Clopen)
{
    DyadicInterval di;

    ASSERT_EQ(di.excluded_end(), 1.0);
}

TEST(DyadicIntervals, test_included_end_Opencl)
{
    DyadicInterval di{IntervalType::Opencl};

    ASSERT_EQ(di.included_end(), 0.0);
}

TEST(DyadicIntervals, test_excluded_end_Opencl)
{
    DyadicInterval di{IntervalType::Opencl};

    ASSERT_EQ(di.excluded_end(), -1.0);
}

TEST(DyadicIntervals, test_dsup_Clopen)
{
    DyadicInterval di;
    Dyadic expected{1, 0};
    ASSERT_TRUE(dyadic_equals(di.dsup(), expected));
}

TEST(DyadicIntervals, test_dinf_Clopen)
{
    DyadicInterval di;
    Dyadic expected{0, 0};
    ASSERT_TRUE(dyadic_equals(di.dinf(), expected));
}

TEST(DyadicIntervals, test_dsup_Opencl)
{
    DyadicInterval di{IntervalType::Opencl};
    Dyadic expected{0, 0};
    ASSERT_TRUE(dyadic_equals(di.dsup(), expected));
}

TEST(DyadicIntervals, test_dinf_Opencl)
{
    DyadicInterval di{IntervalType::Opencl};
    Dyadic expected{-1, 0};
    ASSERT_TRUE(dyadic_equals(di.dinf(), expected));
}

TEST(DyadicIntervals, test_sup_Clopen)
{
    DyadicInterval di;

    ASSERT_EQ(di.sup(), 1.0);
}

TEST(DyadicIntervals, test_inf_Clopen)
{
    DyadicInterval di;

    ASSERT_EQ(di.inf(), 0.0);
}

TEST(DyadicIntervals, test_sup_Opencl)
{
    DyadicInterval di{IntervalType::Opencl};

    ASSERT_EQ(di.sup(), 0.0);
}

TEST(DyadicIntervals, test_inf_Opencl)
{
    DyadicInterval di{IntervalType::Opencl};

    ASSERT_EQ(di.inf(), -1.0);
}

TEST(DyadicIntervals, test_flip_interval_aligned_Clopen)
{

    DyadicInterval di{Dyadic{0, 1}};
    DyadicInterval expected{Dyadic{1, 1}};

    ASSERT_EQ(di.flip_interval(), expected);
}

TEST(DyadicIntervals, test_flip_interval_non_aligned_Clopen)
{

    DyadicInterval di{Dyadic{1, 1}};
    DyadicInterval expected{Dyadic{0, 1}};

    ASSERT_EQ(di.flip_interval(), expected);
}

TEST(DyadicIntervals, test_flip_interval_aligned_Opencl)
{

    DyadicInterval di{Dyadic{0, 1}, IntervalType::Opencl};
    DyadicInterval expected{Dyadic{-1, 1}, IntervalType::Opencl};

    ASSERT_EQ(di.flip_interval(), expected);
}

TEST(DyadicIntervals, test_flip_interval_non_aligned_Opencl)
{

    DyadicInterval di{Dyadic{1, 1}, IntervalType::Opencl};
    DyadicInterval expected{Dyadic{2, 1}, IntervalType::Opencl};

    ASSERT_EQ(di.flip_interval(), expected);
}

TEST(DyadicIntervals, test_aligned_aligned)
{
    DyadicInterval di{Dyadic{0, 0}};

    ASSERT_TRUE(di.aligned());
}

TEST(DyadicIntervals, test_aligned_non_aligned)
{
    DyadicInterval di{Dyadic{1, 0}};

    ASSERT_FALSE(di.aligned());
}

TEST(DyadicIntervals, test_contains_unit_and_half)
{
    DyadicInterval parent{Dyadic{0, 0}}, child{Dyadic{0, 1}};

    ASSERT_TRUE(parent.contains_dyadic(child));
}

TEST(DyadicIntervals, test_contains_unit_and_half_compliment)
{
    DyadicInterval parent{Dyadic{0, 0}}, child{Dyadic{1, 1}};

    ASSERT_TRUE(parent.contains_dyadic(child));
}

TEST(DyadicIntervals, test_contains_unit_and_unit)
{
    DyadicInterval parent{Dyadic{0, 0}}, child{Dyadic{0, 0}};

    ASSERT_TRUE(parent.contains_dyadic(child));
}

TEST(DyadicIntervals, test_contains_unit_and_longer)
{
    DyadicInterval parent{Dyadic{0, 0}}, child{Dyadic{0, -1}};

    ASSERT_FALSE(parent.contains_dyadic(child));
}

TEST(DyadicIntervals, test_contains_unit_disjoint)
{
    DyadicInterval parent{Dyadic{0, 0}}, child{Dyadic{1, 0}};

    ASSERT_FALSE(parent.contains_dyadic(child));
}

TEST(DyadicIntervals, test_contains_unit_disjoint_and_shorter_right)
{
    DyadicInterval parent{Dyadic{0, 0}}, child{Dyadic{2, 1}};

    ASSERT_FALSE(parent.contains_dyadic(child));
}

TEST(DyadicIntervals, test_contains_unit_disjoint_and_shorter_left)
{
    DyadicInterval parent{Dyadic{0, 0}}, child{Dyadic{-1, 1}};

    ASSERT_FALSE(parent.contains_dyadic(child));
}

TEST(DyadicIntervals, test_to_dyadic_intervals_unit_interval_tol_1)
{

    auto intervals = to_dyadic_intervals(RealInterval(0.0, 1.0), 1);
    DyadicInterval expected{Dyadic{0, 0}};

    ASSERT_EQ(intervals.size(), 1);
    ASSERT_EQ(intervals[0], expected);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_unit_interval_tol_5)
{

    auto intervals = to_dyadic_intervals(RealInterval(0.0, 1.0), 5);
    DyadicInterval expected{Dyadic{0, 0}};

    ASSERT_EQ(intervals.size(), 1);
    ASSERT_EQ(intervals[0], expected);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_mone_one_interval_tol_1)
{

    auto intervals = to_dyadic_intervals(RealInterval(-1.0, 1.0), 1);
    DyadicInterval expected0{Dyadic{-1, 0}}, expected1{Dyadic{0, 0}};

    ASSERT_EQ(intervals.size(), 2);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_mone_onehalf_interval_tol_1)
{

    auto intervals = to_dyadic_intervals(RealInterval(-1.0, 1.5), 1);
    DyadicInterval expected0{Dyadic{-1, 0}}, expected1{Dyadic{0, 0}},
            expected2{Dyadic{2, 1}};

    ASSERT_EQ(intervals.size(), 3);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
    ASSERT_EQ(intervals[2], expected2);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_mone_onequarter_interval_tol_1)
{

    auto intervals = to_dyadic_intervals(RealInterval(-1.0, 1.25), 1);
    DyadicInterval expected0{Dyadic{-1, 0}}, expected1{Dyadic{0, 0}};

    ASSERT_EQ(intervals.size(), 2);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_mone_onequarter_interval_tol_2)
{

    auto intervals = to_dyadic_intervals(RealInterval(-1.0, 1.25), 2);
    DyadicInterval expected0{Dyadic{-1, 0}}, expected1{Dyadic{0, 0}},
            expected2{Dyadic{4, 2}};

    ASSERT_EQ(intervals.size(), 3);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
    ASSERT_EQ(intervals[2], expected2);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_0_upper_interval_tol_1)
{
    auto intervals = to_dyadic_intervals(RealInterval(0.0, 1.63451), 1);
    DyadicInterval expected0{Dyadic{0, 0}}, expected1{Dyadic{2, 1}};

    ASSERT_EQ(intervals.size(), 2);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_0_upper_interval_tol_2)
{
    auto intervals = to_dyadic_intervals(RealInterval(0.0, 1.63451), 2);
    DyadicInterval expected0{Dyadic{0, 0}}, expected1{Dyadic{2, 1}};

    ASSERT_EQ(intervals.size(), 2);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_0_upper_interval_tol_3)
{
    auto intervals = to_dyadic_intervals(RealInterval(0.0, 1.63451), 3);
    DyadicInterval expected0{Dyadic{0, 0}}, expected1{Dyadic{2, 1}},
            expected2{Dyadic{12, 3}};

    ASSERT_EQ(intervals.size(), 3);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
    ASSERT_EQ(intervals[2], expected2);
}

TEST(DyadicIntervals, test_to_dyadic_intervals_0_upper_interval_tol_7)
{
    auto intervals = to_dyadic_intervals(RealInterval(0.0, 1.63451), 7);
    DyadicInterval expected0{Dyadic{0, 0}}, expected1{Dyadic{2, 1}},
            expected2{Dyadic{12, 3}}, expected3{Dyadic{208, 7}};

    ASSERT_EQ(intervals.size(), 4);
    ASSERT_EQ(intervals[0], expected0);
    ASSERT_EQ(intervals[1], expected1);
    ASSERT_EQ(intervals[2], expected2);
    ASSERT_EQ(intervals[3], expected3);
}

TEST(DyadicIntervals, shrink_interval_left_Clopen)
{
    DyadicInterval start{Dyadic{0, 0}};

    DyadicInterval expected{Dyadic{0, 1}};

    ASSERT_EQ(start.shrink_interval_left(), expected);
}

TEST(DyadicIntervals, shrink_interval_right_Clopen)
{
    DyadicInterval start{Dyadic{0, 0}};

    DyadicInterval expected{Dyadic{1, 1}};

    ASSERT_EQ(start.shrink_interval_right(), expected);
}

TEST(DyadicIntervals, shrink_interval_left_Opencl)
{
    DyadicInterval start{Dyadic{0, 0}, IntervalType::Opencl};

    DyadicInterval expected{Dyadic{-1, 1}, IntervalType::Opencl};

    ASSERT_EQ(start.shrink_interval_left(), expected);
}

TEST(DyadicIntervals, shrink_interval_right_Opencl)
{
    DyadicInterval start{Dyadic{0, 0}, IntervalType::Opencl};

    DyadicInterval expected{Dyadic{0, 1}, IntervalType::Opencl};

    ASSERT_EQ(start.shrink_interval_right(), expected);
}

TEST(DyadicIntervals, shrink_to_contained_end_Clopen)
{
    DyadicInterval start{Dyadic{0, 0}};

    DyadicInterval expected{Dyadic{0, 1}};

    ASSERT_EQ(start.shrink_to_contained_end(), expected);
}

TEST(DyadicIntervals, shrink_to_omitted_end_Clopen)
{
    DyadicInterval start{Dyadic{0, 0}};

    DyadicInterval expected{Dyadic{1, 1}};

    ASSERT_EQ(start.shrink_to_omitted_end(), expected);
}

TEST(DyadicIntervals, shrink_to_contained_end_Opencl)
{
    DyadicInterval start{Dyadic{0, 0}, IntervalType::Opencl};

    DyadicInterval expected{Dyadic{0, 1}, IntervalType::Opencl};

    ASSERT_EQ(start.shrink_to_contained_end(), expected);
}

TEST(DyadicIntervals, shrink_to_omitted_end_Opencl)
{
    DyadicInterval start{Dyadic{0, 0}, IntervalType::Opencl};

    DyadicInterval expected{Dyadic{-1, 1}, IntervalType::Opencl};

    ASSERT_EQ(start.shrink_to_omitted_end(), expected);
}

TEST(DyadicInterval, TestSerialization)
{
    DyadicInterval indi(143, 295);
    std::stringstream ss;
    {
        archives::JSONOutputArchive oarc(ss);
        oarc(indi);
    }

    DyadicInterval outdi;
    {
        archives::JSONInputArchive iarc(ss);
        iarc(outdi);
    }

    EXPECT_EQ(indi, outdi);
}
