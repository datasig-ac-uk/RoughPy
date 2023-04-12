// Copyright (c) 2023 Datasig Group. All rights reserved.
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
// Created by user on 12/04/23.
//

#include "brownian_stream.h"
#include <gtest/gtest.h>

using namespace rpy;

using rpy::intervals::DyadicInterval;

class BrownianStreamTests : public ::testing::Test {
public:
    static constexpr deg_t width = 2;
    static constexpr deg_t depth = 2;
    static constexpr algebra::VectorType vtype = algebra::VectorType::Dense;
    std::size_t seed = 12345;

    const scalars::ScalarType* ctype;
    algebra::context_pointer ctx;


    class BrownianHolder : public streams::BrownianStream {
    public:
        using streams::BrownianStream::BrownianStream;
    };

    BrownianHolder bm;






    BrownianStreamTests()
        : seed(12345),
          ctype(scalars::ScalarType::of<double>()),
          ctx(algebra::get_context(width, depth, ctype, {})),
          bm(ctype->get_rng("pcg", seed), {width, {0.0, 1.0}, ctx, ctype, vtype, 1})
    {}

};



TEST_F(BrownianStreamTests, TestLogSignatureResolutions) {
    DyadicInterval unit(0, 0);

    auto top = bm.log_signature(unit, 1, *ctx);

    DyadicInterval left(0, 1);
    DyadicInterval right(1, 1);

    auto bottom_left = bm.log_signature(left, 1, *ctx);
    auto bottom_right = bm.log_signature(right, 1, *ctx);

    EXPECT_EQ(top, ctx->cbh(bottom_left, bottom_right, vtype));

}
