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

#ifndef ROUGHPY_ALGEBRA_LINEAR_OPERATOR_H_
#define ROUGHPY_ALGEBRA_LINEAR_OPERATOR_H_

#include "algebra_base.h"
#include "algebra_bundle.h"
#include "algebra_fwd.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace algebra {

template <typename Argument, typename Result>
class LinearOperatorInterface
{

public:
    using argument_type = Argument;
    using result_type = Result;

    virtual ~LinearOperatorInterface();

    virtual result_type eval(const argument_type& arg) const;
};

template <typename Argument, typename Result>
class LinearOperator
{
    using interface_type = LinearOperatorInterface<Argument, Result>;
    boost::intrusive_ptr<interface_type> p_impl;

public:
    Result operator()(const Argument& arg) const;
};

template <typename Argument, typename Result>
Result LinearOperator<Argument, Result>::operator()(const Argument& arg) const
{
    if (p_impl) { return p_impl->eval(arg); }
    return Result();
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_LINEAR_OPERATOR_H_
