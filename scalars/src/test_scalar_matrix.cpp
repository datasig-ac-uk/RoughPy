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
// Created by user on 11/05/23.
//

#include <gtest/gtest.h>

#include <roughpy/scalars/scalar_matrix.h>

#include <sstream>

using namespace rpy;
using namespace rpy::scalars;

TEST(ScalarMatrix, ScalarMatrixSerialization)
{

    std::vector<float> src{1.0F, 2.0F, 3.0F, 4.0F};

    ScalarMatrix in_mat(2, 2,
                        {ScalarType::of<float>(), src.data(), src.size()});
    std::stringstream ss;
    {
        archives::JSONOutputArchive oarc(ss);
        oarc(in_mat);
    }

    ScalarMatrix out_mat;
    {
        archives::JSONInputArchive iarc(ss);
        iarc(out_mat);
    }

    EXPECT_EQ(out_mat.type(), in_mat.type());
    EXPECT_EQ(out_mat.nrows(), in_mat.nrows());
    EXPECT_EQ(out_mat.ncols(), in_mat.ncols());
    EXPECT_EQ(out_mat.layout(), in_mat.layout());
    EXPECT_EQ(out_mat.storage(), in_mat.storage());

    EXPECT_EQ(out_mat.size(), out_mat.nrows() * out_mat.ncols());

    const auto* aptr = in_mat.raw_cast<const float*>();
    const auto* bptr = out_mat.raw_cast<const float*>();

    ASSERT_NE(aptr, nullptr);
    ASSERT_NE(bptr, nullptr);

    for (int i = 0; i < 4; ++i) { ASSERT_EQ(aptr[i], bptr[i]); }
}
