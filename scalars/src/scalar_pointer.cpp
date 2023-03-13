//
// Created by user on 26/02/23.
//

#include "scalar_pointer.h"

#include <stdexcept>

#include "scalar_type.h"
#include "scalar.h"


using namespace rpy::scalars;

void *ScalarPointer::ptr() {
    if (m_constness == IsConst) {
        throw std::runtime_error("attempting to convert const pointer to non-const pointer");
    }
    return const_cast<void*>(p_data);
}
Scalar ScalarPointer::deref() const noexcept {
    return Scalar(*this);
}
Scalar ScalarPointer::deref_mut() {
    if (m_constness == IsConst) {
        throw std::runtime_error("attempting to dereference const pointer to non-const value");
    }
    return Scalar(*this);
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
    return {p_type, static_cast<const void*>(new_ptr), m_constness};
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
