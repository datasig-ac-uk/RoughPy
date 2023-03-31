// Copyright (c) 2023 Datasig Group. All rights reserved.
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

#ifndef ROUGHPY_SCALARS_SCALAR_POINTER_H_
#define ROUGHPY_SCALARS_SCALAR_POINTER_H_

#include "scalars_fwd.h"

#include <roughpy/core/traits.h>


#ifndef RPY_IF_CONSTEXPR
#define RPY_IF_CONSTEXPR
#endif

namespace rpy { namespace scalars {

namespace dtl {
}


class ScalarPointer {
public:
    enum Constness {
        IsConst,
        IsMutable
    };

protected:
    const ScalarType *p_type = nullptr;
    const void *p_data = nullptr;

    Constness m_constness = IsMutable;

public:

    ScalarPointer(const ScalarType *type, const void *data, Constness constness)
        : p_type(type), p_data(data), m_constness(constness) {}

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    ScalarPointer() = default;

    explicit ScalarPointer(const ScalarType *type) : p_type(type) {}

    ScalarPointer(const ScalarType* type, void* ptr)
        : p_type(type), p_data(ptr), m_constness(IsMutable)
    {}
    ScalarPointer(const ScalarType* type, const void* ptr)
        : p_type(type), p_data(ptr), m_constness(IsConst)
    {}




    /**
     * @brief Get whether the pointer is const or not
     * @return bool, true if pointer is const
     */
    bool is_const() const noexcept { return m_constness == IsConst; }

    /**
     * @brief Get whether the pointer is the null pointer
     * @return bool, true if the underlying raw pointer is null
     */
    bool is_null() const noexcept { return p_data == nullptr; }

    /**
     * @brief Get a pointer to the type of this scalar
     * @return pointer to type object
     */
    const ScalarType *type() const noexcept {
        return p_type;
    }

    /**
     * @brief Get the raw pointer contained held
     * @return const raw pointer to underlying data
     */
    const void *ptr() const noexcept { return p_data; }

    /**
     * @brief Get the raw pointer contained held
     * @return const raw pointer to underlying data
     */
    const void *cptr() const noexcept { return p_data; }

    /**
     * @brief Get the mutable raw pointer held
     * @return mutable raw pointer to underlying data
     */
    void *ptr();

    /**
     * @brief Cast the pointer to a const raw type
     * @tparam T Type to cast to
     * @return pointer to underlying type of T
     */
    template <typename T>
    traits::ensure_pointer<T> raw_cast() const noexcept {
        return static_cast<traits::ensure_pointer<T>>(p_data);
    }

    /**
     * @brief Cast the pointer to a raw type
     * @tparam T Type to cast to
     * @return pointer to underlying type of T
     */
    template <typename T, typename=std::enable_if_t<!std::is_const<std::remove_pointer_t<T>>::value>>
    traits::ensure_pointer<T> raw_cast() {
        if (m_constness == IsConst) {
            throw std::runtime_error("cannot cast const pointer to non-const pointer");
        }
        return static_cast<traits::ensure_pointer<T>>(const_cast<void*>(p_data));
    }

    /**
     * @brief Dereference to get a scalar value
     * @return new Scalar object referencing the pointed to data
     */
    Scalar deref() const noexcept;

    /**
     * @brief Dereference to a scalar type mutably
     * @return new Scalar object mutably referencing te pointed to data
     */
    Scalar deref_mut();

    // Pointer-like operations

    constexpr operator bool() const noexcept {
        return p_data != nullptr;
    }

    Scalar operator*();
    Scalar operator*() const noexcept;

    ScalarPointer operator+(size_type index) const noexcept;
    ScalarPointer &operator+=(size_type index) noexcept;

    ScalarPointer &operator++() noexcept;
    const ScalarPointer operator++(int) noexcept;

    Scalar operator[](size_type index) const noexcept;
    Scalar operator[](size_type index);

    difference_type operator-(const ScalarPointer &right) const noexcept;

};

inline bool operator==(const ScalarPointer &left, const ScalarPointer &right) {
    return left.type() == right.type() && left.ptr() == right.ptr();
}

inline bool operator!=(const ScalarPointer &left, const ScalarPointer &right) {
    return left.type() != right.type() || left.ptr() != right.ptr();
}

}}

#endif // ROUGHPY_SCALARS_SCALAR_POINTER_H_
