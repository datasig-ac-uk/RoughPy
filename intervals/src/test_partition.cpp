// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 05/07/23.
//

#include <gtest/gtest.h>

#include <roughpy/intervals/partition.h>

using namespace rpy;
using namespace rpy::intervals;

TEST(Partitions, TestPartitionMeshNoIntermediatePoints) {
    Partition part(RealInterval(0.0, 1.0));

    ASSERT_EQ(part.mesh(), 1.0);
}

TEST(Partitions, TestPartitionsMeshHalvedAfterRefineMidpoints) {
    Partition part(RealInterval(0.0, 1.0));

    auto refined = part.refine_midpoints();

    ASSERT_EQ(refined.mesh(), part.mesh() / 2.0);
}

TEST(Partitions, TestMeshNonUniformIntermediates) {
    param_t midpts[] = {0.3, 0.6};
    Partition parts(RealInterval(0.0, 1.0), midpts);

    EXPECT_EQ(parts.mesh(), 0.3);
}


TEST(Partitions, TestPartitionsRefineMidpointIntermediates) {
    param_t midpts[] = { 0.3, 0.6 };
    Partition parts(RealInterval(0.0, 1.0), midpts);

    auto refined = parts.refine_midpoints();

    EXPECT_EQ(refined.size(), 2*parts.size());

    std::vector<param_t> expected_intermediates {
            0.15, 0.3, 0.45, 0.6, 0.8
    };

    const auto& intermediates = refined.intermediates();
    for (dimn_t i=0; i<refined.size()-1; ++i) {
        EXPECT_DOUBLE_EQ(intermediates[i], expected_intermediates[i]);
    }

}

TEST(Partitions, TestPartitionsRefineMidpointsSize) {
    Partition part(RealInterval(0.0, 1.0));
    ASSERT_EQ(part.size(), 1);

    auto refined1 = part.refine_midpoints();
    ASSERT_EQ(refined1.size(), 2);

    auto refined2 = refined1.refine_midpoints();
    ASSERT_EQ(refined2.size(), 4);
}

TEST(Partitions, TestAccessNoIntermediates) {
    Partition part(RealInterval(0.0, 1.0));

    EXPECT_EQ(part[0], RealInterval(0.0, 1.0));
}

TEST(Partitions, TestAccessOneIntermediate) {
    param_t midpt = 0.5;
    Partition part(RealInterval(0.0, 1.0), midpt);

    EXPECT_EQ(part[0], RealInterval(0.0, midpt));
    EXPECT_EQ(part[1], RealInterval(midpt, 1.0));
}

TEST(Partitions, TestAccessTwoIntermediates) {
    param_t midpts[] = {0.3, 0.6};
    Partition part(RealInterval(0.0, 1.0), midpts);

    EXPECT_EQ(part[0], RealInterval(0.0, midpts[0]));
    EXPECT_EQ(part[1], RealInterval(midpts[0], midpts[1]));
    EXPECT_EQ(part[2], RealInterval(midpts[1], 1.0));
}

TEST(Partitions, TestInsertNewIntermediateNoPoints) {
    Partition parts(RealInterval(0.0, 1.0));

    parts.insert_intermediate(0.5);
    EXPECT_EQ(parts.size(), 2);
    EXPECT_EQ(parts.mesh(), 0.5);
}

TEST(Partitions, TestInsertNewIntermetiateWithIntermediates) {
    param_t midpts[] = { 0.3, 0.6 };
    Partition parts(RealInterval(0.0, 1.0), midpts);

    parts.insert_intermediate(0.5);
    EXPECT_EQ(parts.size(), 4);
    EXPECT_DOUBLE_EQ(parts.mesh(), 0.1);
}

TEST(Partitions, TestMergeTwoEmpty) {
    Partition lparts(RealInterval(0.0, 1.0));
    Partition rparts(RealInterval(0.5, 1.5));

    auto merged = lparts.merge(rparts);


    EXPECT_EQ(merged.size(), 3);
    EXPECT_EQ(merged.inf(), 0.0);
    EXPECT_EQ(merged.sup(), 1.5);
    EXPECT_EQ(merged.mesh(), 0.5);
}

TEST(Partitions, TestMergeNestedEmpty) {
    Partition outer(RealInterval(0.0, 1.0));
    Partition inner(RealInterval(0.3, 0.6));

    auto merged = outer.merge(inner);

    EXPECT_EQ(merged.size(), 3);
    EXPECT_EQ(merged.inf(), 0.0);
    EXPECT_EQ(merged.sup(), 1.0);

    const auto& intermediates = merged.intermediates();
    EXPECT_EQ(intermediates[0], 0.3);
    EXPECT_EQ(intermediates[1], 0.6);
}

TEST(Partitions, TestMergeFullyDisjointEmpty) {
    Partition left(RealInterval(0.0, 1.0));
    Partition right(RealInterval(1.5, 2.5));

    auto merged = left.merge(right);


    EXPECT_EQ(merged.size(), 3);
    EXPECT_EQ(merged.inf(), 0.0);
    EXPECT_EQ(merged.sup(), 2.5);

    const auto& intermediates = merged.intermediates();
    EXPECT_EQ(intermediates[0], 1.0);
    EXPECT_EQ(intermediates[1], 1.5);
}

TEST(Partitions, TestMergePartiallyDisjointEmpty) {
    Partition left(RealInterval(0.0, 1.0));
    Partition right(RealInterval(1.0, 2.5));

    auto merged = left.merge(right);


    EXPECT_EQ(merged.size(), 2);
    EXPECT_EQ(merged.inf(), 0.0);
    EXPECT_EQ(merged.sup(), 2.5);

    const auto& intermediates = merged.intermediates();
    EXPECT_EQ(intermediates[0], 1.0);
}


TEST(Partitions, TestMergeNestedSameInf) {
    Partition left(RealInterval(0.0, 1.0));
    Partition right(RealInterval(0.0, 2.5));

    auto merged = left.merge(right);

    EXPECT_EQ(merged.size(), 2);
    EXPECT_EQ(merged.inf(), 0.0);
    EXPECT_EQ(merged.sup(), 2.5);

    const auto& intermediates = merged.intermediates();
    EXPECT_EQ(intermediates[0], 1.0);
}

TEST(Partitions, TestMergeNestedSameSup) {
    Partition left(RealInterval(0.5, 1.0));
    Partition right(RealInterval(0.0, 1.0));

    auto merged = left.merge(right);

    EXPECT_EQ(merged.size(), 2);
    EXPECT_EQ(merged.inf(), 0.0);
    EXPECT_EQ(merged.sup(), 1.0);

    const auto& intermediates = merged.intermediates();
    EXPECT_EQ(intermediates[0], 0.5);
}

TEST(Partitions, TestMergeNestIntervalMatchesIntermediates) {
    param_t midpts[] = { 0.3, 0.6 };
    Partition outer(RealInterval(0.0, 1.0), midpts);
    Partition inner(RealInterval(midpts[0], midpts[1]));

    auto merged = outer.merge(inner);

    EXPECT_EQ(merged.size(), outer.size());
    EXPECT_EQ(merged.intermediates(), outer.intermediates());
}

TEST(Partitions, TestMergeNestIntervalNonMatchingIntermediates) {
    param_t midpts[] = {0.3, 0.6};
    Partition outer(RealInterval(0.0, 1.0), midpts);
    Partition inner(RealInterval(0.2, 0.4));

    auto merged = outer.merge(inner);

    ASSERT_EQ(merged.size(), 5);

    const auto& intermediates = merged.intermediates();
    EXPECT_EQ(intermediates[0], 0.2);
    EXPECT_EQ(intermediates[1], 0.3);
    EXPECT_EQ(intermediates[2], 0.4);
    EXPECT_EQ(intermediates[3], 0.6);

}

TEST(Partitions, TestMergeEqualWithInterlevedIntermediates) {
    param_t lmidpts[] = { 0.3, 0.6 };
    param_t rmidpts[] = { 0.2, 0.4 };
    Partition left(RealInterval(0.0, 1.0), lmidpts);
    Partition right(RealInterval(0.0, 1.0), rmidpts);

    auto merged = left.merge(right);
    ASSERT_EQ(merged.size(), 5);

    const auto& intermediates = merged.intermediates();
    EXPECT_EQ(intermediates[0], 0.2);
    EXPECT_EQ(intermediates[1], 0.3);
    EXPECT_EQ(intermediates[2], 0.4);
    EXPECT_EQ(intermediates[3], 0.6);
}
