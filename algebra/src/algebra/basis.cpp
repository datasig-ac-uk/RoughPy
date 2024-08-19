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
// Created by user on 02/03/23.
//
#include <roughpy/algebra/basis.h>

#include "vector.h"

using namespace rpy;
using namespace rpy::algebra;

Basis::~Basis() = default;

dimn_t Basis::max_dimension() const noexcept { return 0; }

dimn_t Basis::dense_dimension(dimn_t size) const { return size; }

bool Basis::less(BasisKeyCRef RPY_UNUSED_VAR k1, BasisKeyCRef RPY_UNUSED_VAR k2) const
{
    RPY_THROW(std::runtime_error, "basis is not ordered");
}

dimn_t Basis::to_index(BasisKeyCRef RPY_UNUSED_VAR key) const
{
    RPY_THROW(std::runtime_error, "basis is not ordered");
}

BasisKey Basis::to_key(dimn_t index) const
{
    RPY_THROW(std::runtime_error, "basis is not ordered");
}

KeyRange Basis::iterate_keys() const
{
    RPY_THROW(std::runtime_error, "basis is not ordered");
}

algebra::dtl::BasisIterator Basis::keys_begin() const
{
    return dtl::BasisIterator();
}
algebra::dtl::BasisIterator Basis::keys_end() const
{
    return dtl::BasisIterator();
}

deg_t Basis::max_degree() const
{
    RPY_THROW(std::runtime_error, "basis is not graded");
}

deg_t Basis::degree(BasisKeyCRef RPY_UNUSED_VAR key) const
{
    RPY_THROW(std::runtime_error, "basis is not graded");
}

KeyRange Basis::iterate_keys_of_degree(deg_t degree) const
{
    RPY_THROW(std::runtime_error, "basis is not graded or ordered");
}

deg_t Basis::alphabet_size() const
{
    RPY_THROW(std::runtime_error, "basis is not word-like");
}

bool Basis::is_letter(BasisKeyCRef RPY_UNUSED_VAR key) const
{
    RPY_THROW(std::runtime_error, "basis is not word-like");
}

let_t Basis::get_letter(BasisKeyCRef RPY_UNUSED_VAR key) const
{
    RPY_THROW(std::runtime_error, "basis is not word-like");
}

pair<BasisKey, BasisKey>
Basis::parents(BasisKeyCRef RPY_UNUSED_VAR key) const
{
    RPY_THROW(std::runtime_error, "basis is not word-like");
}
BasisComparison Basis::compare(BasisPointer other) const noexcept
{
    if (other == this) { return BasisComparison::IsSame; }
    return BasisComparison::IsNotCompatible;
}
dimn_t Basis::dimension_to_degree(deg_t degree) const
{
    RPY_THROW(std::runtime_error, "basis is not graded");
}
bool Basis::supports_key_type(const devices::TypePtr& type) const noexcept
{
    return false;
}
Slice<const devices::TypePtr> Basis::supported_key_types() const noexcept
{
    return {};
}
