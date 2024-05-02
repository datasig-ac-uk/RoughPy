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

#include "scalar.h"
#include "scalar/casts.h"
#include "scalar_array_element.h"
#include "scalar_serialization.h"
#include "scalar_type.h"
#include "traits.h"

#include "algorithms.h"
#include "scalar/raw_bytes.h"

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/host_device.h>

#include <cereal/types/vector.hpp>

using namespace rpy;
using namespace scalars;

ScalarArray::ScalarArray() = default;

ScalarArray::ScalarArray(const ScalarArray& other)
    : p_type(other.p_type),
      m_buffer(other.m_buffer)
{}

ScalarArray::ScalarArray(ScalarArray&& other) noexcept
    : p_type(other.p_type),
      m_buffer(std::move(other.m_buffer))
{}

ScalarArray::ScalarArray(const ScalarType* type, dimn_t size) : p_type(type)
{
    RPY_CHECK(type != nullptr);
    m_buffer = devices::Buffer(type->device(), size, type->type_info());
}

ScalarArray::ScalarArray(devices::TypeInfo info, dimn_t size)
    : m_buffer(size, info)
{
    RPY_CHECK(traits::is_fundamental(info));
}

ScalarArray::ScalarArray(PackedScalarType type, const void* data, dimn_t size)
    : m_buffer(data, size, type_info_from(type))
{
    if (type.is_pointer()) { p_type = type.get_pointer(); }
}
ScalarArray::ScalarArray(PackedScalarType type, void* data, dimn_t size)
    : m_buffer(data, size, type_info_from(type))
{
    if (type.is_pointer()) { p_type = type.get_pointer(); }
}
ScalarArray::ScalarArray(PackedScalarType type, devices::Buffer&& buffer)
    : m_buffer(std::move(buffer))
{
    RPY_CHECK(m_buffer.is_null() || type_info_from(type) == m_buffer.type_info());
    if (type.is_pointer()) { p_type = type.get_pointer(); }

    auto dst_device = scalars::device_from(type);
    if (dst_device && (*dst_device) != m_buffer.device()) {}
}

// ScalarArray::ScalarArray(const ScalarType* type, devices::Buffer&& buffer)
//     : p_type(type)
// {
//     RPY_CHECK(type != nullptr);
//     auto source_device = buffer.device();
//     auto target_device = type->device();
//     const auto source_info = buffer.type_info();
//     const auto target_info = type->type_info();
//     if (source_info == target_info) {
//         if (source_device == target_device) {
//             m_buffer = std::move(buffer);
//         } else {
//             buffer.to_device(m_buffer, target_device);
//         }
//     } else {
//         ScalarArray tmp(source_info, std::move(buffer));
//         m_buffer = devices::Buffer(type->device(), tmp.size(), target_info);
//         type->convert_copy(*this, tmp);
//     }
// }
//
// ScalarArray::ScalarArray(devices::TypeInfo info, devices::Buffer&& buffer)
// {
//     const auto source_info = buffer.type_info();
//     if (buffer.type_info() == info) {
//         if (buffer.is_host()) {
//             m_buffer = std::move(buffer);
//         } else {
//             buffer.to_device(m_buffer, devices::get_host_device());
//         }
//     } else {
//         RPY_CHECK(traits::is_fundamental(info));
//         devices::Buffer tmp;
//         if (m_buffer.is_host()) {
//             tmp = std::move(buffer);
//         } else {
//             buffer.to_device(tmp, devices::get_host_device());
//         }
//
//         m_buffer = devices::Buffer(tmp.size(), info);
//
//         if (!dtl::scalar_convert_copy(
//                     m_buffer.ptr(),
//                     info,
//                     tmp.ptr(),
//                     source_info,
//                     tmp.size()
//             )) {
//             RPY_THROW(std::runtime_error, "failed to convert copy into
//             target");
//         }
//     }
// }

ScalarArray::~ScalarArray()
{
    m_buffer.~Buffer();
    p_type = nullptr;
}

ScalarArray& ScalarArray::operator=(const ScalarArray& other)
{
    if (&other != this) {
        this->~ScalarArray();

        const auto info = other.type_info();
        const auto bytes = other.size();

        if (traits::is_fundamental(info)) { RPY_CHECK(p_type != nullptr); }
    }
    return *this;
}

ScalarArray& ScalarArray::operator=(ScalarArray&& other) noexcept
{
    if (&other != this) {
        this->~ScalarArray();
        p_type = other.p_type;
        m_buffer = std::move(other.m_buffer);
    }
    return *this;
}

ScalarArray ScalarArray::copy_or_clone() && { return ScalarArray(); }

PackedScalarType ScalarArray::type() const noexcept
{
    if (p_type != nullptr) { return p_type; }
    return m_buffer.type_info();
}

devices::TypeInfo ScalarArray::type_info() const noexcept
{
    return m_buffer.type_info();
}

// const void* ScalarArray::raw_pointer(dimn_t i) const noexcept
//{
//     const void* ptr = nullptr;
//
//     switch (p_type_and_mode.get_enumeration()) {
//         case dtl::ScalarArrayStorageModel::BorrowConst:
//             ptr = const_borrowed;
//             break;
//         case dtl::ScalarArrayStorageModel::BorrowMut: ptr = mut_borrowed;
//         break; case dtl::ScalarArrayStorageModel::Owned:
//             ptr = owned_buffer.ptr();
//             break;
//     }
//
//     if (ptr == nullptr || i == 0) { return ptr; }
//     const auto info = type_info();
//
//     return static_cast<const byte*>(ptr) + i * info.bytes;
// }

// void* ScalarArray::raw_mut_pointer(dimn_t i) noexcept
//{
//     void* ptr = nullptr;
//
//     switch (p_type_and_mode.get_enumeration()) {
//         case dtl::ScalarArrayStorageModel::BorrowConst: break;
//         case dtl::ScalarArrayStorageModel::BorrowMut: ptr = mut_borrowed;
//         break; case dtl::ScalarArrayStorageModel::Owned:
//             ptr = owned_buffer.ptr();
//             break;
//     }
//
//     if (ptr == nullptr || i == 0) { return ptr; }
//     const auto info = type_info();
//
//     return static_cast<byte*>(ptr) + i * info.bytes;
// }

// const void* ScalarArray::pointer() const
//{
//     switch (p_type_and_mode.get_enumeration()) {
//         case dtl::ScalarArrayStorageModel::BorrowConst: return
//         const_borrowed; case dtl::ScalarArrayStorageModel::BorrowMut: return
//         mut_borrowed; case dtl::ScalarArrayStorageModel::Owned:
//             if (owned_buffer.device() == devices::get_host_device()
//                 && owned_buffer.mode() != devices::BufferMode::Write) {
//                 return owned_buffer.ptr();
//             }
//             RPY_THROW(
//                     std::runtime_error,
//                     "cannot get pointer from devices::Buffer object safely"
//             );
//     }
//     RPY_UNREACHABLE_RETURN(nullptr);
// }

// void* ScalarArray::mut_pointer()
//{
//
//     switch (p_type_and_mode.get_enumeration()) {
//         case dtl::ScalarArrayStorageModel::BorrowConst:
//             RPY_THROW(
//                     std::runtime_error,
//                     "attempting to mutable borrow a const value"
//             );
//         case dtl::ScalarArrayStorageModel::BorrowMut: return mut_borrowed;
//         case dtl::ScalarArrayStorageModel::Owned:
//             if (owned_buffer.device() == devices::get_host_device()
//                 && owned_buffer.mode() != devices::BufferMode::Read) {
//                 return owned_buffer.ptr();
//             }
//             RPY_THROW(
//                     std::runtime_error,
//                     "cannot get pointer from devices::Buffer object safely"
//             );
//     }
//     RPY_UNREACHABLE_RETURN(nullptr);
// }

const devices::Buffer& ScalarArray::buffer() const { return m_buffer; }

devices::Buffer& ScalarArray::mut_buffer() { return m_buffer; }

Scalar ScalarArray::operator[](dimn_t i) const
{
    RPY_CHECK(i < size());
    //    check_for_ptr_access();
    //    RPY_CHECK(i < m_buffer.size());
    //
    //    if (p_type != nullptr) {
    //        return Scalar(p_type_and_mode.get_pointer(), raw_pointer(i));
    //    }
    //    return Scalar(p_type_and_mode.get_type_info(), raw_pointer(i));
    return Scalar(std::make_unique<ScalarArrayElement>(buffer(), i, p_type));
}

Scalar ScalarArray::operator[](dimn_t i)
{
    //    check_for_ptr_access(true);
    //    RPY_CHECK(
    //            i < m_size,
    //            "index " + std::to_string(i) + " out of bounds for array of
    //            size "
    //                    + std::to_string(m_size)
    //    );
    //
    //    if (p_type_and_mode.is_pointer()) {
    //        return Scalar(p_type_and_mode.get_pointer(), raw_mut_pointer(i));
    //    }
    //    return Scalar(p_type_and_mode.get_type_info(), raw_mut_pointer(i));
    RPY_CHECK(i < size());
    return Scalar(std::make_unique<ScalarArrayElement>(mut_buffer(), i, p_type)
    );
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

    return {p_type, m_buffer.slice(offset, sz)};
}

ScalarArray ScalarArray::operator[](SliceIndex index) const
{
    RPY_DBG_ASSERT(index.begin < index.end);
    const auto buffer_size = m_buffer.size();
    RPY_CHECK(
            index.end <= buffer_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(buffer_size)
    );
    const auto offset = index.begin;
    const auto sz = (index.end - index.begin);

    return {p_type, m_buffer.slice(offset, sz)};
}

void ScalarArray::check_for_ptr_access(bool mut) const
{
    RPY_CHECK(m_buffer.is_host());
    RPY_CHECK(!mut || m_buffer.mode() != devices::BufferMode::Read);
}

dimn_t ScalarArray::capacity() const noexcept { return m_buffer.size(); }

containers::Vec<byte> ScalarArray::to_raw_bytes() const
{
    return dtl::to_raw_bytes(m_buffer.ptr(), m_buffer.size(), type());
}

void ScalarArray::from_raw_bytes(
        PackedScalarType type,
        dimn_t count,
        Slice<byte> bytes
)
{
    RPY_CHECK(is_null());
    dtl::from_raw_bytes(m_buffer.ptr(), count, bytes, type);
}

ScalarArray ScalarArray::borrow() const
{
    return ScalarArray(p_type, devices::Buffer(m_buffer));
}

ScalarArray ScalarArray::borrow_mut()
{
    return ScalarArray(p_type, devices::Buffer(m_buffer));
}

ScalarArray ScalarArray::to_device(devices::Device device) const
{
    if (device == this->device()) { return *this; }

    auto new_buffer = device->alloc(type_info(), this->size());
    m_buffer.to_device(new_buffer, device);
    return {p_type, std::move(new_buffer)};
}

void scalars::convert_copy(ScalarArray& dst, const ScalarArray& src)
{
    algorithms::copy(dst, src);
}

#define RPY_EXPORT_MACRO ROUGHPY_SCALARS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME ScalarArray
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>
