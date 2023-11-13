// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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
// Created by sam on 13/03/23.
//

#include "ScalarTests.h"
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_types.h>

#include <vector>

using ScalarTypeTests = rpy::scalars::testing::ScalarTests;
using namespace rpy::scalars;

using rpy::devices::DeviceInfo;
using rpy::devices::DeviceType;

TEST_F(ScalarTypeTests, ConvertCopyFromNonScalarType)
{

    std::vector<std::int32_t> ints{1, 2, 3, 4, 5, 6};

    auto alloced = dtype->allocate(ints.size());

    dtype->convert_copy(alloced, ScalarArray(rpy::Slice<int32_t>(ints)));

    auto raw = alloced.as_slice<const double>();
    for (std::size_t i = 0; i < ints.size(); ++i) {
        ASSERT_EQ(raw[i], static_cast<double>(ints[i]));
    }
}

TEST_F(ScalarTypeTests, CopyConvertFromScalarType)
{
    std::vector<float> floats{1.01, 2.02, 3.03, 4.04, 5.05, 6.06};

    auto alloced = dtype->allocate(floats.size());

    dtype->convert_copy(
            alloced,
            ScalarArray{ftype, floats.data(), floats.size()}
    );

    auto raw = alloced.as_slice<double>();
    for (std::size_t i = 0; i < floats.size(); ++i) {
        ASSERT_EQ(raw[i], static_cast<double>(floats[i]));
    }
}
//
// TEST_F(ScalarTypeTests, ToScalarFromFloat)
// {
//     float val = 1.000001;// doesn't have an exact float representation
//     auto dble = ftype->to_scalar_t({ftype, &val});
//
//     ASSERT_FLOAT_EQ(dble, val);
// }

// TEST_F(ScalarTypeTests, AssignNumAndDenom)
// {
    // double val = 0.0;
    // dtype->assign({dtype, &val}, 1LL, 256LL);

    // ASSERT_DOUBLE_EQ(val, 1.0 / 256.0);
// }
