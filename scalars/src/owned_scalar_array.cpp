// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

//
// Created by user on 28/02/23.
//

#include <roughpy/scalars/owned_scalar_array.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>

using namespace rpy::scalars;

OwnedScalarArray::OwnedScalarArray(const ScalarType* type) : ScalarArray(type)
{}
OwnedScalarArray::OwnedScalarArray(const ScalarArray& other)
    : ScalarArray(other.type()->allocate(other.size()), other.size())
{
    if (other.ptr() != nullptr) {
        p_type->convert_copy(const_cast<void*>(p_data),
                             static_cast<const ScalarPointer&>(other),
                             other.size());
    }
}
OwnedScalarArray::OwnedScalarArray(const OwnedScalarArray& other)
    : ScalarArray(other.type()->allocate(other.size()), other.size())
{
    p_type->convert_copy(const_cast<void*>(p_data),
                         static_cast<const ScalarPointer&>(other),
                         other.size());
}

OwnedScalarArray::OwnedScalarArray(const ScalarType* type, dimn_t size)
    : ScalarArray(type->allocate(size), size)
{}

OwnedScalarArray::OwnedScalarArray(const Scalar& value, dimn_t count)
    : ScalarArray()
{
    const auto* type = value.type();
    if (type != nullptr) {
        ScalarPointer::operator=(type->allocate(count));
        m_size = count;
        type->convert_fill(*this, value.to_pointer(), count, "");
    } else {
        RPY_THROW(std::runtime_error, "scalar value has invalid type");
    }
}

OwnedScalarArray::OwnedScalarArray(const ScalarType* type, const void* data,
                                   dimn_t count)
    : ScalarArray(type)
{
    if (type == nullptr) {
        RPY_THROW(std::invalid_argument, "cannot construct array with invalid type");
    }

    ScalarPointer::operator=(type->allocate(count));
    m_size = count;
    type->convert_copy(const_cast<void*>(p_data), {nullptr, data}, count);
}

OwnedScalarArray::~OwnedScalarArray()
{
    if (p_data != nullptr) {
        OwnedScalarArray::p_type->free(*this, OwnedScalarArray::m_size);
        m_size = 0;
        p_data = nullptr;
    }
}

OwnedScalarArray::OwnedScalarArray(OwnedScalarArray&& other) noexcept
    : ScalarArray(other)
{
    other.p_data = nullptr;// Take ownership
    other.m_size = 0;
}
OwnedScalarArray& OwnedScalarArray::operator=(const ScalarArray& other)
{
    if (&other != this) {
        this->~OwnedScalarArray();
        if (other.size() > 0) {
            ScalarPointer::operator=(other.type()->allocate(other.size()));
            m_size = other.size();
            p_type->convert_copy(const_cast<void*>(p_data), other, m_size);
        } else {
            p_data = nullptr;
            m_size = 0;
            p_type = other.type();
        }
    }
    return *this;
}
OwnedScalarArray& OwnedScalarArray::operator=(OwnedScalarArray&& other) noexcept
{
    if (&other != this) {
        ScalarPointer::operator=(other);
        m_size = other.m_size;
        other.p_data = nullptr;
        other.p_type = nullptr;
        other.m_size = 0;
    }
    return *this;
}
