//
// Created by sam on 20/05/24.
//

#include "value.h"

#include "algorithms.h"

rpy::devices::Value::Value(const Value& other) : p_type(other.p_type)
{
    if (is_inline_stored()) {
        std::memcpy(m_storage.bytes, other.m_storage.bytes, sizeof(void*));
    } else if (p_type != nullptr) {
        m_storage.pointer = p_type->allocate_single();
        p_type->copy(m_storage.pointer, other.m_storage.pointer, 1);
    }
}
rpy::devices::Value::Value(Value&& other) noexcept : p_type(other.p_type)
{
    std::swap(m_storage, other.m_storage);
}

rpy::devices::Value::Value(ConstReference other) : p_type(other.type())
{
    if (is_inline_stored()) {
        std::memcpy(m_storage.bytes, other.data(), size_of(*p_type));
    } else if (p_type != nullptr) {
        m_storage.pointer = p_type->allocate_single();
        p_type->copy(m_storage.pointer, other.data(), 1);
    }
}
rpy::devices::Value::~Value()
{
    if (!is_inline_stored() && m_storage.pointer != nullptr) {
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

            const auto& conversion = p_type->conversions(*other.p_type);

            auto convert = [tp = p_type,
                            ntp = other.p_type,
                            &conversion](auto* dst, const auto* src) {
                if (conversion.convert) {
                    conversion.convert(dst, src);
                    return;
                }

                RPY_THROW(
                        std::runtime_error,
                        string_join(
                                "no valid conversion from type ",
                                tp->id(),
                                " to type ",
                                ntp->id()
                        )
                );
            };

            if (is_inline_stored()) {
                convert(m_storage.bytes, other.data());
            } else {
                if (m_storage.pointer == nullptr) {
                    m_storage.pointer = p_type->allocate_single();
                }
                convert(m_storage.pointer, other.data());
            }
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
            const auto& conversion = p_type->conversions(*other.p_type);

            auto convert = [tp = p_type,
                            ntp = other.p_type,
                            &conversion](auto* dst, auto* src) {
                // We always prefer a move construction if it is possible
                if (conversion.move_convert) {
                    conversion.move_convert(dst, src);
                    return;
                }

                if (conversion.convert) {
                    conversion.convert(dst, src);
                    return;
                }

                RPY_THROW(
                        std::runtime_error,
                        string_join(
                                "no valid conversion from type ",
                                tp->id(),
                                " to type ",
                                ntp->id()
                        )
                );
            };

            if (is_inline_stored()) {
                convert(m_storage.bytes, other.data());
            } else {
                if (m_storage.pointer == nullptr) {
                    m_storage.pointer = p_type->allocate_single();
                }
                convert(m_storage.pointer, other.data());
            }
        }
    }

    return *this;
}

void rpy::devices::Value::change_type(const Type* new_type)
{
    if (p_type == nullptr) {
        p_type = new_type;
    } else if (new_type != p_type) {
        const auto& conversion = new_type->conversions(*p_type);

        auto convert = [tp = p_type,
                        ntp = new_type,
                        &conversion](auto* dst, auto* src) {
            // We always prefer a move construction if it is possible
            if (conversion.move_convert) {
                conversion.move_convert(dst, src);
                return;
            }

            if (conversion.convert) {
                conversion.convert(dst, src);
                return;
            }

            RPY_THROW(
                    std::runtime_error,
                    string_join(
                            "no valid conversion from type ",
                            tp->id(),
                            " to type ",
                            ntp->id()
                    )
            );
        };

        Storage tmp_storage;
        std::memcpy(&tmp_storage, &m_storage, sizeof(Storage));
        if (is_inline_stored(new_type) && is_inline_stored()) {
            convert(m_storage.bytes, tmp_storage.bytes);
        } else if (is_inline_stored(new_type) && !is_inline_stored()) {
            convert(m_storage.bytes, tmp_storage.pointer);
            p_type->free_single(tmp_storage.pointer);
        } else if (is_inline_stored() && !is_inline_stored(new_type)) {
            tmp_storage.pointer = new_type->allocate_single();
            convert(tmp_storage.pointer, tmp_storage.bytes);
        } else {
            tmp_storage.pointer = new_type->allocate_single();
            convert(m_storage.pointer, tmp_storage.pointer);
            p_type->free_single(tmp_storage.pointer);
        }
        p_type = new_type;
    }
}

rpy::devices::Reference&
rpy::devices::Reference::operator=(const ConstReference& other)
{
    if (type() == other.type()) {
        type()->copy(data(), other.data(), 1);
        return *this;
    }

    const auto& conversion = type()->conversions(*other.type());

    if (conversion.convert) {
        conversion.convert(data(), other.data());
        return *this;
    }

    RPY_THROW(
            std::runtime_error,
            string_join(
                    "no valid conversion from type ",
                    type()->id(),
                    " to type ",
                    other.type()->id()
            )
    );

    return *this;
}


rpy::devices::Reference& rpy::devices::Reference::operator=(const Value& other)
{
    if (type() == other.type()) {
        type()->copy(data(), other.data(), 1);
        return *this;
    }

    const auto& conversion = type()->conversions(*other.type());

    if (conversion.convert) {
        conversion.convert(data(), other.data());
        return *this;
    }

    RPY_THROW(
            std::runtime_error,
            string_join(
                    "no valid conversion from type ",
                    type()->id(),
                    " to type ",
                    other.type()->id()
            )
    );

    return *this;
}

rpy::devices::Reference& rpy::devices::Reference::operator=(Value&& other)
{
    if (type() == other.type()) {
        type()->move(data(), other.data(), 1);
    } else {
        const auto& conversion = type()->conversions(*other.p_type);

        auto convert = [tp = type(),
                        ntp = other.p_type,
                        &conversion](auto* dst, auto* src) {
            // We always prefer a move construction if it is possible
            if (conversion.move_convert) {
                conversion.move_convert(dst, src);
                return;
            }

            if (conversion.convert) {
                conversion.convert(dst, src);
                return;
            }

            RPY_THROW(
                    std::runtime_error,
                    string_join(
                            "no valid conversion from type ",
                            tp->id(),
                            " to type ",
                            ntp->id()
                    )
            );
        };

        convert(data(), other.data());
    }

    return *this;
}
