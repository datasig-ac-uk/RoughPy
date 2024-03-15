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
#include "basis_key.h"

using namespace rpy;
using namespace rpy::algebra;


Basis::~Basis() = default;

bool Basis::less(BasisKey RPY_UNUSED_VAR k1, BasisKey RPY_UNUSED_VAR k2) const
{
    return false;
}

dimn_t Basis::to_index(BasisKey RPY_UNUSED_VAR key) const {
    return 0;
}

BasisKey Basis::to_key(dimn_t index) const {
    return BasisKey();
}

KeyRange Basis::iterate_keys() const {
    return KeyRange();
}

deg_t Basis::max_degree() const noexcept
{
    return 0;
}

deg_t Basis::degree(BasisKey RPY_UNUSED_VAR key) const {
    return 0;
}

KeyRange Basis::iterate_keys_of_degree(deg_t degree) const
{
    return KeyRange();
}


deg_t Basis::alphabet_size() const noexcept
{
    return 0;
}

bool Basis::is_letter(BasisKey RPY_UNUSED_VAR key) const {
    return false;
}

let_t Basis::get_letter(BasisKey RPY_UNUSED_VAR key) const {
    return 0;
}

pair<optional<BasisKey>, optional<BasisKey>>
Basis::parents(BasisKey RPY_UNUSED_VAR key) const
{
    return pair<optional<BasisKey>, optional<BasisKey >>();
}


void rpy::algebra::intrusive_ptr_add_ref(const Basis* ptr) noexcept
{
    using ptr_type = boost::intrusive_ref_counter<Basis>;
    intrusive_ptr_add_ref(static_cast<const ptr_type*>(ptr));
}

void rpy::algebra::intrusive_ptr_release(const Basis* ptr) noexcept
{
    using ptr_type = boost::intrusive_ref_counter<Basis>;
    intrusive_ptr_release(static_cast<const ptr_type*>(ptr));
}
