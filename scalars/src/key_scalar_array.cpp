//
// Created by user on 02/03/23.
//

#include "key_scalar_array.h"


#include "scalar_type.h"
#include "scalar.h"
#include "owned_scalar_array.h"



using namespace rpy;
using namespace rpy::scalars;

KeyScalarArray::~KeyScalarArray() {
    if (p_type != nullptr && m_scalars_owned) {
        auto size = ScalarArray::size();
        p_type->free(*this, size);
    }
    if (m_keys_owned) {
        delete[] p_keys;
    }
}

KeyScalarArray::KeyScalarArray(const KeyScalarArray &other)
    : ScalarArray(other.type()->allocate(other.size()), other.size()),
      m_scalars_owned(true) {
    m_constness = IsMutable;
    if (other.p_data != nullptr) {
        p_type->convert_copy(const_cast<void *>(p_data), other, m_size);

        if (p_keys != nullptr) {
            allocate_keys();
            std::copy(other.p_keys, other.p_keys + other.m_size, const_cast<key_type *>(p_keys));
        }
    } else {
        assert(other.p_keys == nullptr);
    }
}
KeyScalarArray::KeyScalarArray(KeyScalarArray &&other) noexcept
    : ScalarArray(other),
      p_keys(other.p_keys),
      m_scalars_owned(other.m_scalars_owned),
      m_keys_owned(other.m_keys_owned) {
    m_constness = IsMutable;
    other.m_scalars_owned = false;
    other.p_keys = nullptr;
    other.p_data = nullptr;
    assert(other.p_data == nullptr);
}
KeyScalarArray::KeyScalarArray(OwnedScalarArray &&sa) noexcept
    : ScalarArray(std::move(sa)), m_scalars_owned(true) {
    m_constness = IsMutable;
}
KeyScalarArray::KeyScalarArray(ScalarArray base, const key_type *keys)
    : ScalarArray(base), p_keys(keys), m_keys_owned(false) {
}
KeyScalarArray::KeyScalarArray(const ScalarType *type) noexcept : ScalarArray(type) {
}
KeyScalarArray::KeyScalarArray(const ScalarType *type, dimn_t n) noexcept
    : ScalarArray(type) {
    allocate_scalars(static_cast<idimn_t>(n));
}

KeyScalarArray::KeyScalarArray(const ScalarType *type, const void *begin, dimn_t count) noexcept
    : ScalarArray(type, begin, count), m_scalars_owned(false) {
}
KeyScalarArray &KeyScalarArray::operator=(const ScalarArray &other) noexcept {
    if (&other != this) {
        this->~KeyScalarArray();
        ScalarArray::operator=(other);
        m_scalars_owned = false;
        m_keys_owned = false;
    }
    return *this;
}
KeyScalarArray &KeyScalarArray::operator=(KeyScalarArray &&other) noexcept {
    if (&other != this) {
        this->~KeyScalarArray();
        m_scalars_owned = other.m_scalars_owned;
        m_keys_owned = other.m_keys_owned;
        p_keys = other.p_keys;
        ScalarArray::operator=(other);
    }
    return *this;
}
KeyScalarArray &KeyScalarArray::operator=(OwnedScalarArray &&other) noexcept {
    m_scalars_owned = true;
    ScalarArray::operator=(std::move(other));
    return *this;
}
void KeyScalarArray::allocate_scalars(idimn_t count) {
    auto new_size = (count == -1) ? m_size : static_cast<dimn_t>(count);
    if (new_size != 0) {
        ScalarArray::operator=({p_type->allocate(new_size), new_size});
        m_scalars_owned = true;
    }
}
void KeyScalarArray::allocate_keys(idimn_t count) {
    auto new_size = (count == -1) ? m_size : static_cast<dimn_t>(count);
    if (p_keys != nullptr && m_keys_owned) {
        delete[] p_keys;
    }
    if (new_size != 0) {
        p_keys = new key_type[new_size];
    } else {
        p_keys = nullptr;
    }
}
key_type *KeyScalarArray::keys() {
    if (m_keys_owned) {
        return const_cast<key_type *>(p_keys);
    }
    throw std::runtime_error("borrowed keys are not mutable");
}

KeyScalarArray::operator OwnedScalarArray() &&noexcept {
    auto result = OwnedScalarArray(p_type, p_data, m_size);
    p_type = nullptr;
    p_data = nullptr;
    m_size = 0;
    return result;
}
