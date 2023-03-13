//
// Created by user on 28/02/23.
//

#include "owned_scalar_array.h"

#include "scalar_type.h"
#include "scalar.h"

using namespace rpy::scalars;

OwnedScalarArray::OwnedScalarArray(const ScalarType *type)
    : ScalarArray(type) {
}
OwnedScalarArray::OwnedScalarArray(const ScalarArray &other)
    : ScalarArray(other.type()->allocate(other.size()), other.size()) {
    if (other.ptr() != nullptr) {
        p_type->convert_copy(const_cast<void *>(p_data),
                             static_cast<const ScalarPointer&>(other),
                             other.size());
    }
}
OwnedScalarArray::OwnedScalarArray(const OwnedScalarArray &other)
    : ScalarArray(other.type()->allocate(other.size()), other.size()) {
    p_type->convert_copy(const_cast<void *>(p_data),
                         static_cast<const ScalarPointer&>(other),
                         other.size());
}

OwnedScalarArray::OwnedScalarArray(const ScalarType *type, dimn_t size)
    : ScalarArray(type->allocate(size), size) {}

OwnedScalarArray::OwnedScalarArray(const Scalar &value, dimn_t count)
    : ScalarArray()
{
    const auto* type = value.type();
    if (type != nullptr) {
        ScalarPointer::operator=(type->allocate(count));
        m_size = count;
        type->convert_fill(*this, value.to_pointer(), count, "");
    } else {
        throw std::runtime_error("scalar value has invalid type");
    }

}

OwnedScalarArray::OwnedScalarArray(const ScalarType *type, const void *data, dimn_t count)
    : ScalarArray(type)
{
    if (type == nullptr) {
        throw std::invalid_argument("cannot construct array with invalid type");
    }

    ScalarPointer::operator=(type->allocate(count));
    m_size = count;
    type->convert_copy(const_cast<void*>(p_data), {nullptr, data}, count);
}

OwnedScalarArray::~OwnedScalarArray() {
    if (p_data != nullptr) {
        OwnedScalarArray::p_type->free(*this, OwnedScalarArray::m_size);
        m_size = 0;
        p_data = nullptr;
    }
}

OwnedScalarArray::OwnedScalarArray(OwnedScalarArray &&other) noexcept
    : ScalarArray(other) {
    other.p_data = nullptr; // Take ownership
    other.m_size = 0;

}
OwnedScalarArray &OwnedScalarArray::operator=(const ScalarArray &other) {
    if (&other != this) {
        this->~OwnedScalarArray();
        if (other.size() > 0) {
            ScalarPointer::operator=(other.type()->allocate(other.size()));
            m_size = other.size();
            p_type->convert_copy(const_cast<void *>(p_data), other, m_size);
        } else {
            p_data = nullptr;
            m_size = 0;
            p_type = other.type();
        }
    }
    return *this;
}
OwnedScalarArray &OwnedScalarArray::operator=(OwnedScalarArray &&other) noexcept {
    if (&other != this) {
        ScalarPointer::operator=(other);
        m_size = other.m_size;
        other.p_data = nullptr;
        other.p_type = nullptr;
        other.m_size = 0;
    }
    return *this;
}
