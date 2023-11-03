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

#ifndef ROUGHPY_SCALARS_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_SCALAR_ARRAY_H_

#include "scalars_fwd.h"
#include "packed_scalar_type_ptr.h"

#include <roughpy/platform/serialization.h>
#include <roughpy/device/buffer.h>

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

class RPY_EXPORT ScalarArray
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

public:
    ScalarArray();

    explicit ScalarArray(const ScalarType* type, dimn_t size = 0);
    explicit ScalarArray(devices::TypeInfo info, dimn_t size = 0);

    explicit ScalarArray(const ScalarType* type, devices::Buffer&& buffer);
    explicit ScalarArray(devices::TypeInfo info, devices::Buffer&& buffer);

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

    bool is_owning() const noexcept {
        return p_type_and_mode.get_enumeration() == discriminator_type::Owned;
    }

    optional<const ScalarType*> type() const noexcept;

    devices::TypeInfo type_info() const noexcept;

    constexpr dimn_t size() const noexcept { return m_size; }

    const void* pointer() const;
    void* mut_pointer();
    const devices::Buffer& buffer() const;
    devices::Buffer& mut_buffer();

    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();


private:
    const void* raw_pointer() const noexcept;
};

template <typename T>
ScalarArray::ScalarArray(Slice<T> data)
    : p_type_and_mode(
            scalar_type_of<T>(),
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
            scalar_type_of<T>(),
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
            scalar_type_of<T>(),
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
            scalar_type_of<T>(),
            dtl::ScalarArrayStorageModel::BorrowConst
    ),
      const_borrowed(data),
      m_size(size)
{
    check_pointer_and_size(data, size);
}

RPY_SERIAL_EXTERN_SAVE_CLS(ScalarArray)
RPY_SERIAL_EXTERN_LOAD_CLS(ScalarArray)


}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_ARRAY_H_
