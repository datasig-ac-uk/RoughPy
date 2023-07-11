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
// Created by user on 02/05/23.
//

#ifndef ROUGHPY_DEVICE_SRC_DEVICE_FREE_TENSOR_H
#define ROUGHPY_DEVICE_SRC_DEVICE_FREE_TENSOR_H

#include "device_algebra_base.h"

#include <roughpy/algebra/free_tensor.h>

namespace rpy {
namespace device {

class DeviceFreeTensor
    : public DeviceAlgebraBase<algebra::FreeTensorInterface, DeviceFreeTensor>
{

    using base_t
            = DeviceAlgebraBase<algebra::FreeTensorInterface, DeviceFreeTensor>;

public:
    DeviceFreeTensor(scalars::ScalarArray&& data, const DeviceContext* ctx)
        : base_t(std::move(data), ctx)
    {}

    void mul_inplace(const algebra_t& other) override;

    algebra_t mul(const algebra_t& other) const override;
    void add_mul(const algebra_t& lhs, const algebra_t& rhs) override;
    void sub_mul(const algebra_t& lhs, const algebra_t& rhs) override;
    void mul_smul(const algebra_t& rhs, const scalars::Scalar& scalar) override;
    void mul_sdiv(const algebra_t& rhs, const scalars::Scalar& scalar) override;
    algebra::FreeTensor exp() const override;
    algebra::FreeTensor log() const override;
    algebra::FreeTensor inverse() const override;
    algebra::FreeTensor antipode() const override;
    void fmexp(const algebra::FreeTensor& other) override;
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_DEVICE_FREE_TENSOR_H
