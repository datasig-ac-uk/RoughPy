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

#ifndef ROUGHPY_SCALARS_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_SCALAR_ARRAY_H_

#include "packed_scalar_type_ptr.h"
#include "scalars_fwd.h"
#include "scalar_type.h"

#include <roughpy/platform/devices/buffer.h>
#include <roughpy/platform/serialization.h>

namespace rpy {
namespace scalars {

namespace dtl {
enum class ScalarArrayStorageModel
{
    BorrowConst = 0,
    BorrowMut = 1,
    Owned = 2,
};

}

struct SliceIndex {
    dimn_t begin;
    dimn_t end;
};

class ROUGHPY_SCALARS_EXPORT ScalarArray
{
    using discriminator_type = dtl::ScalarArrayStorageModel;
    using type_pointer = PackedScalarTypePointer<dtl::ScalarArrayStorageModel>;

    type_pointer p_type_and_mode;

    union
    {
        devices::Buffer owned_buffer;
        const void* const_borrowed;
        void* mut_borrowed;
    };

    dimn_t m_size = 0;

    static bool check_pointer_and_size(const void* ptr, dimn_t size);

protected:
    RPY_NO_DISCARD
    type_pointer packed_type() const noexcept { return p_type_and_mode; }

    ScalarArray(type_pointer type, void* data, dimn_t size)
        : p_type_and_mode(type),
          mut_borrowed(data),
          m_size(size)
    {
        p_type_and_mode.update_enumeration(
                dtl::ScalarArrayStorageModel::BorrowMut
        );
    }
    ScalarArray(type_pointer type, const void* data, dimn_t size)
        : p_type_and_mode(type),
          const_borrowed(data),
          m_size(size)
    {
        p_type_and_mode.update_enumeration(
                dtl::ScalarArrayStorageModel::BorrowConst
        );
    }

public:
    ScalarArray();
    ScalarArray(const ScalarArray& other);
    ScalarArray(ScalarArray&& other) noexcept;

    explicit ScalarArray(const ScalarType* type, dimn_t size = 0);
    explicit ScalarArray(devices::TypeInfo info, dimn_t size = 0);

    ScalarArray(const ScalarType* type, const void* data, dimn_t size);
    ScalarArray(devices::TypeInfo info, const void* data, dimn_t size);

    ScalarArray(const ScalarType* type, void* data, dimn_t size);
    ScalarArray(devices::TypeInfo info, void* data, dimn_t size);

    ScalarArray(const ScalarType* type, devices::Buffer&& buffer);
    ScalarArray(devices::TypeInfo info, devices::Buffer&& buffer);

    template <typename T>
    explicit ScalarArray(Slice<T> data);

    template <typename T>
    explicit ScalarArray(Slice<const T> data);

    template <typename T>
    ScalarArray(T* data, dimn_t size);

    template <typename T>
    ScalarArray(const T* adata, dimn_t size);

    ~ScalarArray();

    ScalarArray& operator=(const ScalarArray& other);
    ScalarArray& operator=(ScalarArray&& other) noexcept;

    ScalarArray copy_or_clone() &&;

    RPY_NO_DISCARD
    bool is_owning() const noexcept
    {
        return p_type_and_mode.get_enumeration() == discriminator_type::Owned;
    }

    RPY_NO_DISCARD
    optional<const ScalarType*> type() const noexcept;

    RPY_NO_DISCARD
    devices::TypeInfo type_info() const noexcept;

    RPY_NO_DISCARD
    constexpr dimn_t size() const noexcept { return m_size; }
    RPY_NO_DISCARD
    dimn_t capacity() const noexcept;
    RPY_NO_DISCARD
    constexpr bool empty() const noexcept { return m_size == 0; }
    RPY_NO_DISCARD
    constexpr bool is_null() const noexcept
    {
        return p_type_and_mode.is_null() && empty();
    }
    RPY_NO_DISCARD
    constexpr bool is_const() const noexcept
    {
        return p_type_and_mode.get_enumeration()
                == discriminator_type::BorrowConst;
    }
    RPY_NO_DISCARD
    devices::Device device() const noexcept;

    RPY_NO_DISCARD
    const void* pointer() const;
    RPY_NO_DISCARD
    void* mut_pointer();
    RPY_NO_DISCARD
    const devices::Buffer& buffer() const;
    RPY_NO_DISCARD
    devices::Buffer& mut_buffer();

    RPY_NO_DISCARD
    Scalar operator[](dimn_t i) const;
    RPY_NO_DISCARD
    Scalar operator[](dimn_t i);

    RPY_NO_DISCARD
    ScalarArray operator[](SliceIndex index);
    RPY_NO_DISCARD
    ScalarArray operator[](SliceIndex index) const;

    ScalarArray borrow() const;
    ScalarArray borrow_mut();

    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();

private:
    void check_for_ptr_access(bool mut = false) const;
    RPY_NO_DISCARD
    std::vector<byte> to_raw_bytes() const;
    void from_raw_bytes(devices::TypeInfo info, dimn_t count, Slice<byte> bytes);

public:
    template <typename T>
    RPY_NO_DISCARD Slice<T> as_mut_slice()
    {
        check_for_ptr_access(true);
        return {static_cast<T*>(raw_mut_pointer()), m_size};
    }

    template <typename T>
    RPY_NO_DISCARD Slice<const T> as_slice() const
    {
        check_for_ptr_access(false);
        return {static_cast<const T*>(raw_pointer()), m_size};
    }

private:
    RPY_NO_DISCARD
    const void* raw_pointer(dimn_t i = 0) const noexcept;
    RPY_NO_DISCARD
    void* raw_mut_pointer(dimn_t i = 0) noexcept;
};

template <typename T>
ScalarArray::ScalarArray(Slice<T> data)
    : p_type_and_mode(
              devices::type_info<T>(),
              dtl::ScalarArrayStorageModel::BorrowMut
      ),
      mut_borrowed(data.data()),
      m_size(data.size())
{
    check_pointer_and_size(data.data(), data.size());
}

template <typename T>
ScalarArray::ScalarArray(Slice<const T> data)
    : p_type_and_mode(
              devices::type_info<T>(),
              dtl::ScalarArrayStorageModel::BorrowConst
      ),
      const_borrowed(data.data()),
      m_size(data.size())
{
    check_pointer_and_size(data.data(), data.size());
}

template <typename T>
ScalarArray::ScalarArray(T* data, dimn_t size)
    : p_type_and_mode(
              devices::type_info<T>(),
              dtl::ScalarArrayStorageModel::BorrowMut
      ),
      mut_borrowed(data),
      m_size(size)
{
    check_pointer_and_size(data, size);
}
template <typename T>
ScalarArray::ScalarArray(const T* data, dimn_t size)
    : p_type_and_mode(
              devices::type_info<T>(),
              dtl::ScalarArrayStorageModel::BorrowConst
      ),
      const_borrowed(data),
      m_size(size)
{
    check_pointer_and_size(data, size);
}


RPY_SERIAL_LOAD_FN_IMPL(ScalarArray)
{
    devices::TypeInfo type_info;
    RPY_SERIAL_SERIALIZE_VAL(type_info);
    uint64_t count;
    RPY_SERIAL_SERIALIZE_VAL(count);
    std::vector<byte> raw_bytes;
    RPY_SERIAL_SERIALIZE_VAL(raw_bytes);
    from_raw_bytes(type_info, count, raw_bytes);
}
RPY_SERIAL_SAVE_FN_IMPL(ScalarArray)
{
    RPY_SERIAL_SERIALIZE_NVP("type_info", type_info());
    RPY_SERIAL_SERIALIZE_NVP("count", static_cast<uint64_t>(size()));
    RPY_SERIAL_SERIALIZE_NVP("raw_bytes", to_raw_bytes());
}

#ifdef RPY_COMPILING_SCALARS
RPY_SERIAL_EXTERN_SAVE_CLS_BUILD(ScalarArray)
RPY_SERIAL_EXTERN_LOAD_CLS_BUILD(ScalarArray)
#else
RPY_SERIAL_EXTERN_SAVE_CLS_IMP(ScalarArray)
RPY_SERIAL_EXTERN_LOAD_CLS_IMP(ScalarArray)
#endif

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_ARRAY_H_
