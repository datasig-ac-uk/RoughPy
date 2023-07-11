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

#include <roughpy/streams/lie_increment_stream.h>

#include <gtest/gtest.h>
#include <random>

#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar_pointer.h>

using namespace rpy;
using namespace rpy::streams;

namespace rpy {
namespace algebra {

inline void PrintTo(const Lie& arg, std::ostream* os) { arg.print(*os); }

inline void PrintTo(const FreeTensor& arg, std::ostream* os) { arg.print(*os); }

}// namespace algebra
}// namespace rpy

namespace {

class RandomScalars
{
    std::mt19937 rng;
    std::normal_distribution<double> dist;

public:
    const scalars::ScalarType* ctype;

    explicit RandomScalars(double std_dev)
        : rng(std::random_device()()), dist(0.0, std_dev),
          ctype(scalars::ScalarType::of<double>())
    {}

    scalars::OwnedScalarArray random_data(dimn_t count)
    {
        std::vector<double> tmp_data;

        tmp_data.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            tmp_data.push_back(dist(rng));
        }

        scalars::OwnedScalarArray result(ctype, count);
        scalars::ScalarPointer src(ctype, tmp_data.data());
        ctype->convert_copy(result.ptr(), src, count);

        return result;
    }
};

class LieIncrementStreamTests : public ::testing::Test
{
    RandomScalars gen;

public:
    static constexpr deg_t width = 5;
    static constexpr deg_t depth = 2;
    algebra::context_pointer ctx;

    StreamMetadata md{width, intervals::RealInterval(0.0, 1.0), ctx,
                      ctx->ctype(), algebra::VectorType::Dense};

    LieIncrementStreamTests()
        : gen(1.0), ctx(algebra::get_context(width, depth, gen.ctype,
                                             {{"backend", "libalgebra_lite"}}))
    {}

    scalars::OwnedScalarArray random_data(dimn_t rows, dimn_t cols = width)
    {
        return gen.random_data(rows * cols);
    }

    std::vector<param_t> indices(deg_t num_increments) const
    {
        if (num_increments == 1) { return {0.0}; }

        std::vector<param_t> result;
        result.reserve(num_increments);

        param_t step(1.0 / (num_increments - 1));
        for (auto i = 0; i < num_increments; ++i) {
            result.push_back(i * step);
        }
        return result;
    }
};

}// namespace

TEST_F(LieIncrementStreamTests, TestLogSignatureSingleIncrement)
{

    auto data = random_data(1);
    algebra::VectorConstructionData edata{
            scalars::KeyScalarArray(scalars::OwnedScalarArray(data)),
            algebra::VectorType::Dense};
    auto idx = indices(1);
    const streams::LieIncrementStream path(
            scalars::KeyScalarArray(std::move(data)), idx, md);

    auto ctx1 = ctx->get_alike(1);
    auto lsig = path.log_signature(intervals::RealInterval(0.0, 1.0), 1, *ctx1);

    auto expected = ctx1->construct_lie(edata);

    ASSERT_EQ(lsig, expected) << (lsig - expected);
}

TEST_F(LieIncrementStreamTests, TestLogSignatureTwoIncrementsDepth1)
{

    auto data = random_data(2);

    algebra::VectorConstructionData edata{scalars::KeyScalarArray(ctx->ctype()),
                                          algebra::VectorType::Dense};
    edata.data.allocate_scalars(width);
    edata.data.type()->convert_copy(edata.data.ptr(), data, width);
    for (int i = 0; i < width; ++i) { edata.data[i] += data[i + width]; }

    auto idx = indices(2);
    const LieIncrementStream path(scalars::KeyScalarArray(std::move(data)), idx,
                                  md);

    auto ctx1 = ctx->get_alike(1);

    auto lsig = path.log_signature(intervals::RealInterval(0.0, 2.0), 1, *ctx1);

    auto expected = ctx1->construct_lie(edata);

    ASSERT_EQ(lsig, expected);
}
