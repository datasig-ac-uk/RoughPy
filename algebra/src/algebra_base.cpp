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
// Created by user on 03/03/23.
//

#include <roughpy/algebra/algebra_base.h>

#include <ostream>

#include <roughpy/algebra/context.h>

using namespace rpy;
using namespace rpy::algebra;

algebra::dtl::AlgebraInterfaceBase::AlgebraInterfaceBase(
        context_pointer&& ctx,
        VectorType vtype,
        const scalars::ScalarType* stype,
        ImplementationType impl_type,
        AlgebraType alg_type
)
    : p_ctx(std::move(ctx)), p_coeff_type(stype), m_vector_type(vtype),
      m_impl_type(impl_type), m_alg_type(alg_type)
{}

algebra::dtl::AlgebraInterfaceBase::~AlgebraInterfaceBase() = default;

void rpy::algebra::dtl::print_empty_algebra(std::ostream& os) { os << "{ }"; }

const rpy::scalars::ScalarType*
rpy::algebra::dtl::context_to_scalars(context_pointer const& ptr)
{
    return ptr->ctype();
}

UnspecifiedAlgebraType rpy::algebra::dtl::try_create_new_empty(
        context_pointer ctx, AlgebraType alg_type
)
{
    return ctx->construct(alg_type, {});
}

UnspecifiedAlgebraType algebra::dtl::construct_dense_algebra(
        scalars::ScalarArray&& data, const context_pointer& ctx,
        AlgebraType atype
)
{
    VectorConstructionData cdata{
            {std::move(data), nullptr},
            VectorType::Dense
    };
    return ctx->construct(atype, cdata);
}

void rpy::algebra::dtl::check_contexts_compatible(
        const context_pointer& ref, const context_pointer& other
)
{
    if (ref == other) { return; }

    RPY_CHECK(ref->check_compatible(*other));
}
