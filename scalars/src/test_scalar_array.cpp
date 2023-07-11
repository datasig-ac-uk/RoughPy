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
// Created by user on 10/05/23.
//

#include <gtest/gtest.h>

#include <sstream>

#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/scalar_array.h>

using namespace rpy;
using namespace rpy::scalars;

TEST(ScalarArray, TestScalarArraySerialize)
{
    std::vector<float> nums{1.0F, 2.0F, 3.0F};
    ScalarArray sa(ScalarType::of<float>(), nums.data(), 3);
    std::stringstream ss;
    {
        archives::JSONOutputArchive out_ar(ss);
        out_ar(sa);
    }

    ScalarArray sb;
    {
        archives::JSONInputArchive in_ar(ss);
        in_ar(sb);
    }

    ASSERT_EQ(sa.size(), sb.size());
    ASSERT_EQ(sa.type(), sb.type());
    const auto* aptr = sa.raw_cast<const float*>();
    const auto* bptr = sb.raw_cast<const float*>();

    for (int i = 0; i < 3; ++i) { ASSERT_EQ(aptr[i], bptr[i]); }

    // Deserialization of a raw ScalarArray always results
    // in an owned object, but ScalarArray will never delete
    // its contents upon destruction.
    sb.type()->free(sb, sb.size());
}
