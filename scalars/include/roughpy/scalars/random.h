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

#ifndef ROUGHPY_SCALARS_RANDOM_H_
#define ROUGHPY_SCALARS_RANDOM_H_

#include "owned_scalar_array.h"
#include "scalar_type.h"
#include <roughpy/core/helpers.h>

namespace rpy {
namespace scalars {

class RPY_EXPORT RandomGenerator
{
protected:
    const ScalarType* p_type;

public:
    explicit RandomGenerator(const ScalarType* type) : p_type(type) {}

    virtual ~RandomGenerator() = default;

    virtual void set_seed(Slice<uint64_t> seed_data) = 0;
    virtual void set_state(string_view state) = 0;

    RPY_NO_DISCARD
    virtual std::vector<uint64_t> get_seed() const = 0;
    RPY_NO_DISCARD
    virtual std::string get_type() const = 0;
    RPY_NO_DISCARD
    virtual std::string get_state() const = 0;

    RPY_NO_DISCARD
    virtual OwnedScalarArray uniform_random_scalar(ScalarArray lower,
                                                   ScalarArray upper,
                                                   dimn_t count) const
            = 0;
    RPY_NO_DISCARD
    virtual OwnedScalarArray normal_random(Scalar loc, Scalar scale,
                                           dimn_t count) const
            = 0;
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_RANDOM_H_
