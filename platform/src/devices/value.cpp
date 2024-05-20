//
// Created by sam on 20/05/24.
//


#include "value.h"

#include "algorithms.h"

rpy::devices::Value::Value(const Value& other)
    : p_type(other.p_type)
{
    if (is_inline_stored()) {
        std::memcpy(m_storage.bytes, other.m_storage.bytes, sizeof(void*));
    } else if (p_type != nullptr) {
        p_type->copy(m_storage.pointer, other.m_storage.pointer, 1);
    }
}
rpy::devices::Value::Value(Value&& other) noexcept : p_type(other.p_type)
{
    if (is_inline_stored()) {
        std::memcpy(m_storage.bytes, other.m_storage.bytes, sizeof(void*));
    } else if (p_type != nullptr) {
        p_type->move(m_storage.pointer, other.m_storage.pointer, 1);
    }
}

rpy::devices::Value::Value(ConstReference other)
    : p_type(other.type())
{
    if (is_inline_stored()) {
        std::memcpy(m_storage.bytes, other.data(), 1);
    } else if (p_type != nullptr) {
        p_type->copy(m_storage.pointer, other.data(), 1);
    }
}
rpy::devices::Value::~Value()
{
    if (!is_inline_stored()) {
        RPY_DBG_ASSERT(p_type != nullptr);
        p_type->free_single(m_storage.pointer);
    }
}
rpy::devices::Value& rpy::devices::Value::operator=(const Value& other)
{
    if (&other != this) {
        if (p_type == nullptr || p_type == other.p_type) {
            construct_inplace(this, other);
        } else {

        }
    }

    return *this;
}
rpy::devices::Value& rpy::devices::Value::operator=(Value&& other) noexcept
{
     if (&other != this) {
        if (p_type == nullptr || p_type == other.p_type) {
            construct_inplace(this, std::move(other));
        } else {
        }
    }

    return *this;
}
