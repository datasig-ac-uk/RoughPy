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

#include "scalar.h"
#include "scalar/casts.h"
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
ScalarArray::ScalarArray()
    : const_borrowed(nullptr), m_size(0)
{}
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
            owned_buffer = other.owned_buffer;
            break;
    }
}
ScalarArray::ScalarArray(ScalarArray&& other) noexcept
    : p_type_and_mode(other.p_type_and_mode),
      m_size(other.m_size)
{
    switch(p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
            const_borrowed = other.const_borrowed;
            other.const_borrowed = nullptr;
            break;
        case dtl::ScalarArrayStorageModel::BorrowMut:
            mut_borrowed = other.mut_borrowed;
            other.mut_borrowed = nullptr;
            break;
        case dtl::ScalarArrayStorageModel::Owned:
            owned_buffer = std::move(other.owned_buffer);
            break;
    }
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
ScalarArray::ScalarArray(const ScalarType* type, const void* data, dimn_t size)
    : p_type_and_mode(type, dtl::ScalarArrayStorageModel::BorrowConst),
      const_borrowed(data), m_size(size)
{}
ScalarArray::ScalarArray(devices::TypeInfo info, const void* data, dimn_t size)
    : p_type_and_mode(info, dtl::ScalarArrayStorageModel::BorrowConst),
      const_borrowed(data), m_size(size)
{}
ScalarArray::ScalarArray(const ScalarType* type, void* data, dimn_t size)
    : p_type_and_mode(type, dtl::ScalarArrayStorageModel::BorrowMut),
      mut_borrowed(data), m_size(size)
{}
ScalarArray::ScalarArray(devices::TypeInfo info, void* data, dimn_t size)
    : p_type_and_mode(info, dtl::ScalarArrayStorageModel::BorrowMut),
      mut_borrowed(data), m_size(size)
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

    if (ptr == nullptr) { return ptr; }
    auto info = type_info();

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

    if (ptr == nullptr) { return ptr; }
    auto info = type_info();

    return static_cast<byte*>(ptr) + i*info.bytes;
}

const void* ScalarArray::pointer() const
{
    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst: return const_borrowed;
        case dtl::ScalarArrayStorageModel::BorrowMut: return mut_borrowed;
        case dtl::ScalarArrayStorageModel::Owned:
            RPY_THROW(
                    std::runtime_error,
                    "cannot get pointer from devices::Buffer object safely"
            );
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}
void* ScalarArray::mut_pointer()
{
    RPY_CHECK(
            p_type_and_mode.get_enumeration()
            == dtl::ScalarArrayStorageModel::BorrowMut
    );
    return mut_borrowed;
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
Scalar ScalarArray::operator[](dimn_t i) {
    check_for_ptr_access(true);
    RPY_CHECK(i < m_size);


    if (p_type_and_mode.is_pointer()) {
        return Scalar(p_type_and_mode.get_pointer(), raw_mut_pointer(i));
    }
    return Scalar(p_type_and_mode.get_type_info(), raw_mut_pointer(i));
}

inline void ScalarArray::check_for_ptr_access(bool mut) const {
    RPY_CHECK(!p_type_and_mode.is_pointer() || p_type_and_mode->is_cpu());

    RPY_CHECK(!mut || p_type_and_mode.get_enumeration() !=
            dtl::ScalarArrayStorageModel::BorrowConst);

}
dimn_t ScalarArray::capacity() const noexcept {
    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
        case dtl::ScalarArrayStorageModel::BorrowMut:
            return m_size;
        case dtl::ScalarArrayStorageModel::Owned:
            return owned_buffer.size();
    }
    RPY_UNREACHABLE_RETURN(0);
}
devices::Device ScalarArray::device() const noexcept
{
    switch (p_type_and_mode.get_enumeration()) {
        case dtl::ScalarArrayStorageModel::BorrowConst:
        case dtl::ScalarArrayStorageModel::BorrowMut:
            return devices::get_host_device();
        case dtl::ScalarArrayStorageModel::Owned:
            return owned_buffer.device();
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}

RPY_SERIAL_LOAD_FN_IMPL(rpy::scalars::ScalarArray) {}
RPY_SERIAL_SAVE_FN_IMPL(rpy::scalars::ScalarArray) {}
