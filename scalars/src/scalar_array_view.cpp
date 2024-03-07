// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



#include "scalar_array_view.h"
#include "scalar_array.h"
#include "scalar.h"

#include <roughpy/platform/devices/memory_view.h>
#include <roughpy/platform/devices/buffer.h>


using namespace rpy;
using namespace rpy::scalars;

ScalarArrayView::ScalarArrayView(ScalarArray& array)
    : m_view(array.mut_buffer().map()),
      p_type_and_mode(array.packed_type()),
      m_size(array.size()) {}

optional<const ScalarType*> ScalarArrayView::type() const noexcept
{
    if (p_type_and_mode.is_pointer()) { return p_type_and_mode.get_pointer(); }
    return scalar_type_of(p_type_and_mode.get_type_info());
}

Scalar ScalarArrayView::operator[](dimn_t i) const noexcept
{
    auto info = type_info_from(p_type_and_mode);
    return Scalar(info, m_view.raw_ptr(i * info.bytes));
}

ScalarArrayView ScalarArrayView::operator[](SliceIndex i) const
{
    auto info = type_info_from(p_type_and_mode);
    auto size = i.end - i.begin;
    return {m_view.slice(i.begin * info.bytes, size * info.bytes),
            p_type_and_mode, size};
}
