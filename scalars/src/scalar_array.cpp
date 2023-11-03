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

#include "scalar_array.h"
#include "scalar_type.h"
#include "traits.h"

#include <roughpy/device/host_device.h>

using namespace rpy;
using namespace scalars;

bool ScalarArray::check_pointer_and_size(const void* ptr, dimn_t size)
{
    if (size > 0) { RPY_CHECK(ptr != nullptr); }
    return true;
}
ScalarArray::ScalarArray(const ScalarType* type, dimn_t size)
{
    RPY_DBG_ASSERT(type != nullptr);
    *this = type->allocate(size);
}
ScalarArray::ScalarArray(devices::TypeInfo info, dimn_t size)
    : p_type_and_mode(info, dtl::ScalarArrayStorageModel::Owned),
      m_size(size)
{
    RPY_CHECK(traits::is_fundamental(info));
    if (size != 0) {
        owned_buffer
                = devices::get_host_device()->raw_alloc(size, info.alignment);
    }
}
ScalarArray::ScalarArray(const ScalarType* type, devices::Buffer&& buffer)
    : p_type_and_mode(type, dtl::ScalarArrayStorageModel::Owned),
      owned_buffer(std::move(buffer)),
      m_size(owned_buffer.size())
{}
ScalarArray::ScalarArray(devices::TypeInfo info, devices::Buffer&& buffer)
    : p_type_and_mode(info, dtl::ScalarArrayStorageModel::Owned),
      owned_buffer(std::move(buffer)),
      m_size(owned_buffer.size())
{}

ScalarArray::~ScalarArray()
{
    if (p_type_and_mode.get_enumeration()
        == dtl::ScalarArrayStorageModel::Owned) {
        owned_buffer.~Buffer();
    }
}

ScalarArray& ScalarArray::operator=(const ScalarArray& other) {
    if (&other != this) {
        this->~ScalarArray();

    }
    return *this;
}

ScalarArray& ScalarArray::operator=(ScalarArray&& other) noexcept
{




    return *this;
}

optional<const ScalarType*> ScalarArray::type() const noexcept
{
    if (p_type_and_mode.is_pointer()) { return p_type_and_mode.get_pointer(); }

    return scalar_type_of(p_type_and_mode.get_type_info());
}
devices::TypeInfo ScalarArray::type_info() const noexcept
{
    if (p_type_and_mode.is_pointer()) { return p_type_and_mode->type_info(); }
    return p_type_and_mode.get_type_info();
}
