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

#include "roughpy/core/check.h"                      // for throw_exception
#include "roughpy/core/debug_assertion.h"            // for RPY_DBG_ASSERT

#include "scalar.h"
#include "scalar/casts.h"
#include "scalar_serialization.h"
#include "scalar_type.h"
#include "traits.h"

#include "scalar/raw_bytes.h"

#include <roughpy/platform/devices/buffer.h>
#include <roughpy/platform/devices/host_device.h>

#include <cereal/types/vector.hpp>

using namespace rpy;
using namespace scalars;

bool ScalarArray::check_pointer_and_size(const void* ptr, dimn_t size)
{
    if (size > 0) { RPY_CHECK(ptr != nullptr); }
    return true;
}

ScalarArray::ScalarArray() : const_borrowed(nullptr), m_size(0) {}

ScalarArray::ScalarArray(const ScalarArray& other)
    : p_type_and_mode(other.p_type_and_mode),
      m_size(other.m_size)
{
    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
            const_borrowed = other.const_borrowed;
            break;
        case dtl::ScalarArrayStorageModel::BorrowMut:
            mut_borrowed = other.mut_borrowed;
            break;
        case dtl::ScalarArrayStorageModel::Owned:
            construct_inplace(&owned_buffer, other.owned_buffer);
            break;
    }
}

ScalarArray::ScalarArray(ScalarArray&& other) noexcept
    : p_type_and_mode(other.p_type_and_mode),
      m_size(other.m_size)
{
    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
            const_borrowed = other.const_borrowed;
            other.const_borrowed = nullptr;
            break;
        case dtl::ScalarArrayStorageModel::BorrowMut:
            mut_borrowed = other.mut_borrowed;
            other.mut_borrowed = nullptr;
            break;
        case dtl::ScalarArrayStorageModel::Owned:
            construct_inplace(&owned_buffer, std::move(other.owned_buffer));
            break;
    }
}

ScalarArray::ScalarArray(const ScalarType* type, dimn_t size)
    : p_type_and_mode(type, dtl::ScalarArrayStorageModel::BorrowConst),
      const_borrowed(nullptr)
{
    RPY_DBG_ASSERT(type != nullptr);
    if (size > 0) {
        *this = type->allocate(size);
        RPY_DBG_ASSERT(p_type_and_mode.get_enumeration() == dtl::ScalarArrayStorageModel::Owned);
        RPY_DBG_ASSERT(p_type_and_mode.is_pointer() && p_type_and_mode.get_pointer() == type);
        RPY_DBG_ASSERT(m_size*type->type_info().bytes == owned_buffer.size());
    }
}

ScalarArray::ScalarArray(devices::TypeInfo info, dimn_t size)
    : p_type_and_mode(info, dtl::ScalarArrayStorageModel::Owned),
      m_size(size)
{
    RPY_CHECK(traits::is_fundamental(info));
    if (size != 0) {
        owned_buffer
                = devices::get_host_device()->raw_alloc(size*info.alignment, info.alignment);
    }
}

ScalarArray::ScalarArray(const ScalarType* type, const void* data, dimn_t size)
    : p_type_and_mode(type, dtl::ScalarArrayStorageModel::BorrowConst),
      const_borrowed(data),
      m_size(size)
{}

ScalarArray::ScalarArray(devices::TypeInfo info, const void* data, dimn_t size)
    : p_type_and_mode(info, dtl::ScalarArrayStorageModel::BorrowConst),
      const_borrowed(data),
      m_size(size)
{}

ScalarArray::ScalarArray(const ScalarType* type, void* data, dimn_t size)
    : p_type_and_mode(type, dtl::ScalarArrayStorageModel::BorrowMut),
      mut_borrowed(data),
      m_size(size)
{}

ScalarArray::ScalarArray(devices::TypeInfo info, void* data, dimn_t size)
    : p_type_and_mode(info, dtl::ScalarArrayStorageModel::BorrowMut),
      mut_borrowed(data),
      m_size(size)
{}

ScalarArray::ScalarArray(const ScalarType* type, devices::Buffer&& buffer)
    : p_type_and_mode(type, dtl::ScalarArrayStorageModel::Owned),
      owned_buffer(std::move(buffer)),
      m_size(owned_buffer.size() / type->type_info().bytes)
{}

ScalarArray::ScalarArray(devices::TypeInfo info, devices::Buffer&& buffer)
    : p_type_and_mode(info, dtl::ScalarArrayStorageModel::Owned),
      owned_buffer(std::move(buffer)),
      m_size(owned_buffer.size() / info.bytes)
{}

ScalarArray::~ScalarArray()
{
    if (p_type_and_mode.get_enumeration()
        == dtl::ScalarArrayStorageModel::Owned) {
        owned_buffer.~Buffer();
    }
    const_borrowed = nullptr;
    p_type_and_mode = type_pointer();
}

ScalarArray& ScalarArray::operator=(const ScalarArray& other)
{
    if (&other != this && !other.p_type_and_mode.is_null()) {
        this->~ScalarArray();

        if (p_type_and_mode.is_null()) {
            p_type_and_mode = other.p_type_and_mode;
        }

        auto info = type_info();
        if (p_type_and_mode.is_pointer()) {
            auto device = p_type_and_mode->device();
            owned_buffer = device->raw_alloc(
                    other.size() * info.bytes,
                    info.alignment
            );
            m_size = other.size();
            p_type_and_mode->convert_copy(*this, other);
        } else {
            RPY_CHECK(traits::is_fundamental(info));
            auto device = devices::get_host_device();

            m_size = other.size();
            auto nbytes = m_size * info.bytes;
            owned_buffer = device->raw_alloc(nbytes, info.alignment);

            if (other.p_type_and_mode.is_pointer()) {
                other.p_type_and_mode->convert_copy(*this, other);
            } else {
                auto other_type_info = other.type_info();
                RPY_CHECK(traits::is_fundamental(other_type_info));
                if (info == other_type_info) {
                    std::memcpy(
                            owned_buffer.ptr(),
                            other.raw_pointer(0),
                            nbytes
                    );
                } else {
                    if (!dtl::scalar_convert_copy(
                                owned_buffer.ptr(),
                                info,
                                other.raw_pointer(0),
                                other_type_info,
                                m_size
                        )) {
                        RPY_THROW(
                                std::runtime_error,
                                "unable to convert scalar array values"
                        );
                    }
                }
            }
        }
    }
    return *this;
}

ScalarArray& ScalarArray::operator=(ScalarArray&& other) noexcept
{
    if (&other != this) {
        this->~ScalarArray();
        p_type_and_mode = other.p_type_and_mode;
        m_size = other.m_size;
        switch (p_type_and_mode.get_enumeration()) {
            case dtl::ScalarArrayStorageModel::BorrowMut:
                mut_borrowed = other.mut_borrowed;
                break;
            case dtl::ScalarArrayStorageModel::BorrowConst:
                const_borrowed = other.const_borrowed;
                break;
            case dtl::ScalarArrayStorageModel::Owned:
                owned_buffer = std::move(other.owned_buffer);
                break;
        }
    }
    return *this;
}

ScalarArray ScalarArray::copy_or_clone() && { return ScalarArray(); }

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

const void* ScalarArray::raw_pointer(dimn_t i) const noexcept
{
    const void* ptr = nullptr;

    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
            ptr = const_borrowed;
            break;
        case dtl::ScalarArrayStorageModel::BorrowMut: ptr = mut_borrowed; break;
        case dtl::ScalarArrayStorageModel::Owned:
            ptr = owned_buffer.ptr();
            break;
    }

    if (ptr == nullptr || i == 0) { return ptr; }
    const auto info = type_info();

    return static_cast<const byte*>(ptr) + i * info.bytes;
}

void* ScalarArray::raw_mut_pointer(dimn_t i) noexcept
{
    void* ptr = nullptr;

    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst: break;
        case dtl::ScalarArrayStorageModel::BorrowMut: ptr = mut_borrowed; break;
        case dtl::ScalarArrayStorageModel::Owned:
            ptr = owned_buffer.ptr();
            break;
    }

    if (ptr == nullptr || i == 0) { return ptr; }
    const auto info = type_info();

    return static_cast<byte*>(ptr) + i * info.bytes;
}

const void* ScalarArray::pointer() const
{
    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst: return const_borrowed;
        case dtl::ScalarArrayStorageModel::BorrowMut: return mut_borrowed;
        case dtl::ScalarArrayStorageModel::Owned:
            if (owned_buffer.device() == devices::get_host_device()
                && owned_buffer.mode() != devices::BufferMode::Write) {
                return owned_buffer.ptr();
            }
            RPY_THROW(
                    std::runtime_error,
                    "cannot get pointer from devices::Buffer object safely"
            );
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}

void* ScalarArray::mut_pointer()
{

    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
            RPY_THROW(
                    std::runtime_error,
                    "attempting to mutable borrow a const value"
            );
        case dtl::ScalarArrayStorageModel::BorrowMut: return mut_borrowed;
        case dtl::ScalarArrayStorageModel::Owned:
            if (owned_buffer.device() == devices::get_host_device()
                && owned_buffer.mode() != devices::BufferMode::Read) {
                return owned_buffer.ptr();
            }
            RPY_THROW(
                    std::runtime_error,
                    "cannot get pointer from devices::Buffer object safely"
            );
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}

const devices::Buffer& ScalarArray::buffer() const
{
    RPY_CHECK(
            p_type_and_mode.get_enumeration()
            == dtl::ScalarArrayStorageModel::Owned
    );
    return owned_buffer;
}

devices::Buffer& ScalarArray::mut_buffer()
{
    RPY_CHECK(
            p_type_and_mode.get_enumeration()
            == dtl::ScalarArrayStorageModel::Owned
    );
    return owned_buffer;
}

Scalar ScalarArray::operator[](dimn_t i) const
{
    check_for_ptr_access();
    RPY_CHECK(i < m_size);

    if (p_type_and_mode.is_pointer()) {
        return Scalar(p_type_and_mode.get_pointer(), raw_pointer(i));
    }
    return Scalar(p_type_and_mode.get_type_info(), raw_pointer(i));
}

Scalar ScalarArray::operator[](dimn_t i)
{
    check_for_ptr_access(true);
    RPY_CHECK(
            i < m_size,
            "index " + std::to_string(i) + " out of bounds for array of size "
                    + std::to_string(m_size)
    );

    if (p_type_and_mode.is_pointer()) {
        return Scalar(p_type_and_mode.get_pointer(), raw_mut_pointer(i));
    }
    return Scalar(p_type_and_mode.get_type_info(), raw_mut_pointer(i));
}

ScalarArray ScalarArray::operator[](SliceIndex index)
{
    RPY_DBG_ASSERT(index.begin < index.end);
    RPY_CHECK(
            index.end <= m_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(m_size)
    );

    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::Owned:
            if (owned_buffer.device() == devices::get_host_device()) {
                return {p_type_and_mode,
                        raw_mut_pointer(index.begin),
                        index.end - index.begin};
            }
            RPY_THROW(std::runtime_error, "not implemented");
        case dtl::ScalarArrayStorageModel::BorrowConst:
            RPY_THROW(
                    std::runtime_error,
                    "attempting to borrow const array as muable"
            );
        case dtl::ScalarArrayStorageModel::BorrowMut:
            return {p_type_and_mode,
                    raw_mut_pointer(index.begin),
                    index.end - index.begin};
    }
    RPY_UNREACHABLE_RETURN({});
}

ScalarArray ScalarArray::operator[](SliceIndex index) const
{
    RPY_DBG_ASSERT(index.begin < index.end);
    RPY_CHECK(
            index.end <= m_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(m_size)
    );

    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::Owned:
            if (owned_buffer.device() == devices::get_host_device()) {
                return {p_type_and_mode,
                        raw_pointer(index.begin),
                        index.end - index.begin};
            }
            RPY_THROW(std::runtime_error, "not implemented");
        case dtl::ScalarArrayStorageModel::BorrowConst:
        case dtl::ScalarArrayStorageModel::BorrowMut:
            return {p_type_and_mode,
                    raw_pointer(index.begin),
                    index.end - index.begin};
    }
    RPY_UNREACHABLE_RETURN({});
}

void ScalarArray::check_for_ptr_access(bool mut) const
{
    RPY_CHECK(!p_type_and_mode.is_pointer() || p_type_and_mode->is_cpu());

    RPY_CHECK(
            !mut
            || p_type_and_mode.get_enumeration()
                    != dtl::ScalarArrayStorageModel::BorrowConst
    );
}

dimn_t ScalarArray::capacity() const noexcept
{
    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
        case dtl::ScalarArrayStorageModel::BorrowMut:
            return m_size * type_info().bytes;
        case dtl::ScalarArrayStorageModel::Owned: return owned_buffer.size();
    }
    RPY_UNREACHABLE_RETURN(0);
}

devices::Device ScalarArray::device() const noexcept
{
    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
        case dtl::ScalarArrayStorageModel::BorrowMut:
            return devices::get_host_device();
        case dtl::ScalarArrayStorageModel::Owned: return owned_buffer.device();
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}

std::vector<byte> ScalarArray::to_raw_bytes() const
{
    return dtl::to_raw_bytes(pointer(), m_size, type_info());
}

void ScalarArray::from_raw_bytes(
        devices::TypeInfo info,
        dimn_t count,
        Slice<byte> bytes
)
{
    RPY_CHECK(is_null());
    auto tp_o = scalar_type_of(info);
    if (tp_o) {
        *this = (*tp_o)->allocate(count);
    } else {
        p_type_and_mode
                = type_pointer(info, dtl::ScalarArrayStorageModel::Owned);
        owned_buffer
                = devices::get_host_device()->raw_alloc(count, info.alignment);
    }

    dtl::from_raw_bytes(owned_buffer.ptr(), count, bytes, info);
}

ScalarArray ScalarArray::borrow() const
{
    return ScalarArray(p_type_and_mode, pointer(), m_size);
}

ScalarArray ScalarArray::borrow_mut()
{
    return ScalarArray(p_type_and_mode, mut_pointer(), m_size);
}

#define RPY_EXPORT_MACRO ROUGHPY_SCALARS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME ScalarArray
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>
