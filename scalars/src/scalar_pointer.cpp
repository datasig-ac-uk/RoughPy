// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

//
// Created by user on 26/02/23.
//

#include "scalar_pointer.h"

#include <stdexcept>

#include "scalar_type.h"
#include "scalar.h"


using namespace rpy::scalars;

void *ScalarPointer::ptr() {
    if (is_const()) {
        throw std::runtime_error("attempting to convert const pointer to non-const pointer");
    }
    return const_cast<void*>(p_data);
}
Scalar ScalarPointer::deref() const noexcept {
    return Scalar(*this, (m_flags & ~owning_flag) | constness_flag) ;
}
Scalar ScalarPointer::deref_mut() {
    if (is_const()) {
        throw std::runtime_error("attempting to dereference const pointer to non-const value");
    }
    return Scalar(*this, m_flags & ~owning_flag);
}
Scalar ScalarPointer::operator*() {
    return deref_mut();
}
Scalar ScalarPointer::operator*() const noexcept {
    return deref();
}
ScalarPointer ScalarPointer::operator+(ScalarPointer::size_type index) const noexcept {
    if (p_data == nullptr || p_type == nullptr) {
        return {};
    }

    const auto* new_ptr = static_cast<const char*>(p_data) + index*p_type->itemsize();
    return {p_type, static_cast<const void*>(new_ptr), m_flags & ~owning_flag};
}
ScalarPointer &ScalarPointer::operator+=(ScalarPointer::size_type index) noexcept {
    if (p_data != nullptr && p_type != nullptr) {
        p_data = static_cast<const char*>(p_data) + index * p_type->itemsize();
    }
    return *this;
}
ScalarPointer& ScalarPointer::operator++() noexcept {
    if (p_type != nullptr && p_data != nullptr) {
        p_data = static_cast<const char*>(p_data) + p_type->itemsize();
    }
    return *this;
}
const ScalarPointer ScalarPointer::operator++(int) noexcept {
    ScalarPointer result(*this);
    this->operator++();
    return result;
}
Scalar ScalarPointer::operator[](ScalarPointer::size_type index) const noexcept {
    return (*this + index).deref();
}
Scalar ScalarPointer::operator[](ScalarPointer::size_type index) {
    return (*this + index).deref_mut();
}
ScalarPointer::difference_type ScalarPointer::operator-(const ScalarPointer &right) const noexcept {
    const ScalarType* type = p_type;
    if (type == nullptr){
        if (right.p_type != nullptr) {
            type = right.p_type;
        } else {
            return 0;
        }
    }
    return static_cast<difference_type>(
               static_cast<const char*>(p_data) - static_cast<const char*>(right.p_data)
            ) / type->itemsize();
}
