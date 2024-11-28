//
// Created by sam on 14/11/24.
//

#include "roughpy/generics/values.h"

#include <cstring>


#include "roughpy/generics/comparison_trait.h"
#include "roughpy/generics/number_trait.h"
#include "roughpy/generics/type.h"


using namespace rpy;
using namespace rpy::generics;

void Value::allocate_data()
{
    RPY_DBG_ASSERT(!is_inline_stored());
    RPY_CHECK(p_type);

    auto* old_ptr = m_storage.reset(p_type->allocate_object());
    // Most of the time, this function will be called when the object is newly
    // created, and so it's unlikely that we actually need to free the old data.
    // However, there are a few instances where it might be necessary to
    // replace the data completely with another object.
    if (RPY_UNLIKELY(old_ptr != nullptr)) { p_type->free_object(old_ptr); }
}

void Value::copy_assign_value(const Type* type, const void* source_data)
{
    RPY_CHECK_NE(type, nullptr);

    if (!p_type) {
        p_type = type;
    }

    if (*p_type == *type) {
        const bool is_inline = is_inline_stored();
        if (!is_inline && data() == nullptr) {
            allocate_data();
        }

        auto* dst = data();
        p_type->copy_or_fill(dst, source_data, 1, false);
    } else {
        const auto from = p_type->convert_from(*type);
        if (!from) {
            RPY_THROW(std::invalid_argument, "cannot convert");
        }

        // TODO: impl from
        from->unsafe_convert(data(), source_data, false);
    }

}

void Value::move_assign_value(const Type* type, void* source_data)
{
    RPY_CHECK_NE(type, nullptr);

    if (!p_type) {
        p_type = type;
    }

    if (*p_type == *type) {
        const bool is_inline = is_inline_stored();
        if (!is_inline && data() == nullptr) {
            allocate_data();
        }

        auto* dst = data();
        p_type->move(dst, source_data, 1, false);
    } else {
        const auto from = p_type->convert_from(*type);
        if (!from) {
            RPY_THROW(std::invalid_argument, "cannot convert");
        }

        // TODO: impl from
        from->unsafe_convert(data(), source_data, false);
    }

}



Value::Value() = default;
Value::Value(const Value& other) : p_type(other.p_type)
{
    if (p_type) {
        if (other.is_inline_stored()) {
            void* dst_ptr = m_storage.data(p_type.get());
            const void* src_ptr = other.m_storage.data(p_type.get());
            std::memcpy(dst_ptr, src_ptr, sizeof(void*));
        } else {
            // this time we have to allocate first,
            allocate_data();
            void* dst_ptr = m_storage.data(p_type.get());
            const void* src_ptr = other.m_storage.data(p_type.get());
            p_type->copy_or_fill(dst_ptr, src_ptr, 1, false);
        }
    }
}

void Value::ensure_constructed(const Type* backup_type)
{
    if (!is_valid()) {
        if (backup_type != nullptr) {
            p_type = backup_type;
        } else {
            RPY_THROW(std::runtime_error, "no valid type to assign");
        }
    } else if (backup_type != nullptr) {
        RPY_CHECK_EQ(*p_type, *backup_type);
    }

    if (!is_inline_stored() && m_storage.data(p_type.get()) == nullptr) {
        allocate_data();
    }
}

Value::Value(Value&& other) noexcept
    : p_type(std::move(other.p_type)),
      m_storage(std::move(other.m_storage))
{}

Value::Value(TypePtr type, const void* data)
    : p_type(std::move(type))
{
    RPY_CHECK(p_type);
    if (!is_inline_stored()) { allocate_data(); }
    if (data != nullptr) {
        copy_assign_value(type.get(), data);
    }
}

Value::Value(TypePtr type, string_view str_data)
    : p_type(std::move(type))
{
    RPY_DBG_ASSERT(p_type != nullptr);
    ensure_constructed(type.get());
    if (!p_type->parse_from_string(data(), str_data))
    {
        RPY_THROW(std::invalid_argument, "cannot parse string");
    }
}

Value::Value(ConstRef other)
    : p_type(&other.type())
{
    if (p_type) {
        if (!is_inline_stored()) { allocate_data(); }

        const auto* src_ptr = other.data();
        auto* dst_ptr = m_storage.data(p_type.get());

        p_type->copy_or_fill(dst_ptr, src_ptr, 1, false);
    }
}

Value::~Value()
{
    if (p_type && !is_inline_stored()) {
        p_type->free_object(m_storage.data(p_type.get()));
    }
    m_storage = dtl::ValueStorage();
}

Value& Value::operator=(const Value& other)
{
    if (&other == this) {
        return *this;
    }

    if (!other.p_type) {
        if (p_type) {
            RPY_THROW(std::invalid_argument, "cannot assign invalid value");
        }
        return *this;
    }

    copy_assign_value(other.p_type.get(), other.data());
    return *this;
}

Value& Value::operator=(Value&& other) // NOLINT(*-noexcept-move-constructor)
{
    if (&other == this) {
        return *this;
    }

    if (!p_type) {
        // Type is not set, perform a normal move assignment
        p_type = std::move(other.p_type);
        m_storage = std::move(other.m_storage);
        return *this;
    }

    // If the other type is not set, this is not a valid assignment
    if (!other.p_type) {
        RPY_THROW(std::invalid_argument, "cannot assign invalid value");
    }

    if (p_type == other.p_type) {
        // We can move if available.

        if (is_inline_stored()) {
            std::memcpy(data(), other.data(), sizeof(void*));
        } else {
            m_storage = std::move(other.m_storage);
        }
        return *this;
    }

    // Types are different, so move assignment is off the table.
    const auto from = p_type->convert_from(*other.p_type);
    if (!from) {
        RPY_THROW(std::invalid_argument, "cannot convert");
    }
    return *this;
}

Value& Value::operator=(ConstRef other)
{
    if (RPY_UNLIKELY(!other.is_valid() && !p_type)) {
        RPY_THROW(std::invalid_argument, "cannot assign invalid value");
    }

    copy_assign_value(&other.type(), other.data());
    return *this;
}

Value Value::from_string(TypePtr type, string_view str)
{
    return Value(std::move(type), str);
}