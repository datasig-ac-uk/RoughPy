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

#include "scalar_array.h"

#include <roughpy/core/container/vector.h>

#include "algorithms.h"

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/host_device.h>

#include <cereal/types/vector.hpp>

using namespace rpy;
using namespace scalars;

ScalarArray::ScalarArray() = default;

ScalarArray::ScalarArray(const ScalarArray& other) : Buffer(other) {}

ScalarArray::ScalarArray(ScalarArray&& other) noexcept
    : Buffer(std::move(other))
{}

ScalarArray::ScalarArray(TypePtr type, dimn_t size)
    : Buffer(type, size, devices::get_host_device())
{}

ScalarArray::ScalarArray(TypePtr type, const void* data, dimn_t size)
    : Buffer(type, data, size)
{}
ScalarArray::ScalarArray(TypePtr type, void* data, dimn_t size)
    : Buffer(type, data, size)
{}

const devices::Buffer& ScalarArray::buffer() const { return *this; }

devices::Buffer& ScalarArray::mut_buffer()
{
    RPY_CHECK(!is_const());
    return *this;
}

ScalarCRef ScalarArray::operator[](dimn_t i) const
{
    RPY_CHECK(i < size() && is_host());
    const TypePtr tp = type();
    const auto* p = static_cast<const byte*>(ptr()) + i * size_of(tp);
    return ScalarCRef(p, std::move(tp));
}

ScalarRef ScalarArray::operator[](dimn_t i)
{
    RPY_CHECK(i < size() && is_host() && !is_const());
    const TypePtr tp = type();
    auto* p = static_cast<byte*>(ptr()) + i * size_of(tp);
    return ScalarRef(p, std::move(tp));
}

ScalarArray ScalarArray::operator[](SliceIndex index)
{
    RPY_DBG_ASSERT(index.begin < index.end);
    const auto buffer_size = size();
    RPY_CHECK(
            index.end <= buffer_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(buffer_size)
    );

    const auto offset = index.begin;
    const auto sz = (index.end - index.begin);

    return {slice(offset, sz)};
}

ScalarArray ScalarArray::operator[](SliceIndex index) const
{
    RPY_DBG_ASSERT(index.begin <= index.end);
    const auto buffer_size = size();
    RPY_CHECK(
            index.end <= buffer_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(buffer_size)
    );
    const auto offset = index.begin;
    const auto sz = (index.end - index.begin);

    return {slice(offset, sz)};
}

void ScalarArray::check_for_ptr_access(bool mut) const
{
    RPY_CHECK(is_host());
    RPY_CHECK(!mut || mode() != devices::BufferMode::Read);
}

containers::Vec<byte> ScalarArray::to_raw_bytes() const
{
    // return dtl::to_raw_bytes(ptr(), size(), type());
    return {};
}

void ScalarArray::from_raw_bytes(TypePtr type, dimn_t count, Slice<byte> bytes)
{
    // RPY_CHECK(is_null());
    // dtl::from_raw_bytes(ptr(), count, bytes, type);
}

ScalarArray ScalarArray::borrow() const { return *this; }

ScalarArray ScalarArray::borrow_mut()
{
    RPY_CHECK(!is_const());
    return *this;
}

ScalarArray ScalarArray::to_device(devices::Device device) const
{
    if (device == this->device()) { return *this; }
    auto new_buffer = device->alloc(this->type(), this->size());
    Buffer::to_device(new_buffer, device);
    return {std::move(new_buffer)};
}

void scalars::convert_copy(ScalarArray& dst, const ScalarArray& src)
{
    devices::algorithms::copy(dst, src);
}

#define RPY_EXPORT_MACRO ROUGHPY_SCALARS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME ScalarArray
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>
