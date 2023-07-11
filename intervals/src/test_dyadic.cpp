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

#include <roughpy/intervals/dyadic.h>

#include <sstream>

using namespace rpy;
using namespace rpy::intervals;

TEST(Dyadictests, test_rebase_dyadic_1)
{
    Dyadic val{1, 0};
    val.rebase(1);
    ASSERT_TRUE(dyadic_equals(val, Dyadic{2, 1}));
}

TEST(Dyadictests, test_rebase_dyadic_5)
{
    Dyadic val{1, 0};
    val.rebase(5);
    ASSERT_TRUE(dyadic_equals(val, Dyadic{32, 5}));
}

TEST(Dyadic, TestSerialization)
{
    Dyadic ind(35, 128);
    std::stringstream ss;
    {
        archives::JSONOutputArchive oarc(ss);
        oarc(ind);
    }

    Dyadic outd;
    {
        archives::JSONInputArchive iarc(ss);
        iarc(outd);
    }

    EXPECT_TRUE(dyadic_equals(ind, outd));
}
