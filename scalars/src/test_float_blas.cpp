// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 18/04/23.
//

#include "float_blas.h"
#include <gtest/gtest.h>
#include <roughpy/scalars/scalar_type.h>

#include <vector>

using namespace rpy;
using namespace rpy::scalars;

namespace {

class FloatBlasTests : public ::testing::Test
{
    const ScalarType* ctype;
    std::vector<float> a_data;
    std::vector<float> b_data;

public:
    ScalarMatrix matA;
    ScalarMatrix matB;

    std::unique_ptr<BlasInterface> blas;

    FloatBlasTests()
        : ctype(ScalarType::of<float>()), a_data{1.0F, 1.0F, 2.0F, 3.0F},
          b_data{-1.0F, 2.0F, -2.0F, 3.0F},
          matA(2, 2, ScalarArray(ctype, a_data.data(), 4)),
          matB(2, 2, ScalarArray(ctype, b_data.data(), 4)),
          blas(new FloatBlas(ctype))
    {}
};

}// namespace

TEST_F(FloatBlasTests, TestRowMajorByRowMajorFull)
{

    auto result = blas->matrix_matrix(matA, matB);

    /*
     * [[1.0, 1.0]  [[-1.0, 2.0]   =  [[-3.0,  5.0]
     *  [2.0, 3.0]]  [-2.0, 3.0]]      [-8.0, 13.0]]
     */

    auto* ptr = result.raw_cast<float*>();

    EXPECT_EQ(ptr[0], -3.0F);
    EXPECT_EQ(ptr[1], 5.0F);
    EXPECT_EQ(ptr[2], -8.0F);
    EXPECT_EQ(ptr[3], 13.0F);
}
