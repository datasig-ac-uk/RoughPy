// Copyright (c) 2024 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_SRC_TENSOR_FIXTURE_H
#define ROUGHPY_ALGEBRA_SRC_TENSOR_FIXTURE_H

#include <gtest/gtest.h>

#include "tensor_fixture_context.h"

#include "roughpy/core/ranges.h"
#include "roughpy/core/types.h"
#include "roughpy/algebra/context.h"
#include "roughpy/scalars/scalar_types.h"

namespace rpy {
namespace algebra {
namespace testing {

//! Base fixture for free tensor tests
class TensorFixture : public ::testing::Test
{
protected:
    std::unique_ptr<TensorFixtureContext> builder;

protected:
    void SetUp() override;

public:
    //! Pretty formatting for tensor inequality assertions using gtest
    void ASSERT_TENSOR_EQ(
        const FreeTensor& result,
        const FreeTensor& expected
    ) const;
};

} // namespace testing
} // namespace algebra
} // namespace rpy

#endif // ROUGHPY_ALGEBRA_SRC_TENSOR_FIXTURE_H
