//
// Created by sam on 16/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_METHODS_H
#define ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_METHODS_H

#include "builtin_type.h"

#include <algorithm>
#include <charconv>
#if ((defined(RPY_COMPILER_CLANG)                                              \
      && RPY_COMPILER_CLANG < RPY_COMPILER_VERSION(14, 0, 0))                  \
     || defined(RPY_PLATFORM_MACOS))
#include <sstream>
#endif

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/hash.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/alloc.h"

#include "conversion_factory.h"

#include "builtin_type_ids.h"

namespace rpy::generics {

template <typename T>
string_view BuiltinTypeBase<T>::id() const noexcept
{
    return type_id_of<T>;
}

template <typename T>
const std::type_info& BuiltinTypeBase<T>::type_info() const noexcept
{
    return typeid(T);
}

template <typename T>
void BuiltinTypeBase<T>::inc_ref() const noexcept
{
    // no-op for builtin types
}

template <typename T>
bool BuiltinTypeBase<T>::dec_ref() const noexcept
{
    // return false so it is never destroyed
    return false;
}

template <typename T>
bool BuiltinTypeBase<T>::parse_from_string(
        void* data,
        string_view str
) const noexcept
{
    RPY_DBG_ASSERT_NE(data, nullptr);

    T value;

    const auto* begin = str.data();
    const auto* end = str.data() + str.size();

    std::from_chars_result result;
    if constexpr (is_floating_point_v<T>) {
#if ((defined(RPY_COMPILER_CLANG)                                              \
      && RPY_COMPILER_CLANG < RPY_COMPILER_VERSION(14, 0, 0))                  \
     || defined(RPY_PLATFORM_MACOS))
        // Clang <14.0 and AppleClang don't have support for from_chars
        // for floating_point values
        std::istringstream ss(result);
        ss >> value;
        if (ss.fail()) {
            result.ptr = nullptr;
            result.ec = std::errc::invalid_argument;
        } else {
            result.ptr = end;
            result.ec = std::errc();
        }
#else
        result = std::from_chars(begin, end, value, std::chars_format::general);
#endif
    } else {
        result = std::from_chars(begin, end, value, 10);
    }

    if (result.ec == std::errc::invalid_argument) { return false; }

    RPY_DBG_ASSERT_EQ(result.ptr, end);

    *static_cast<T*>(data) = std::move(value);
    return true;
}

template <typename T>
void* BuiltinTypeBase<T>::allocate_object() const
{
    // TODO: replace with small object allocator
    return mem::aligned_alloc(alignof(T), alignof(T));
}

template <typename T>
void BuiltinTypeBase<T>::free_object(void* ptr) const
{
    // TODO: replace with small object allocator
    mem::aligned_free(ptr);
}

template <typename T>
void BuiltinTypeBase<T>::copy_or_move(
        void* dst,
        const void* src,
        size_t count,
        bool RPY_UNUSED_VAR(move)// Move semantics makes no difference for T
) const noexcept
{

    if (RPY_UNLIKELY(dst == nullptr || count == 0)) { return; }

    auto* dst_ptr = static_cast<T*>(dst);

    if (src == nullptr) {
        // Source is null, which means we should fill the range
        // with 0
        std::fill_n(dst_ptr, count, static_cast<T>(0));
    } else {
        // Source is not null, copy data from src to dst
        const auto* src_ptr = static_cast<const T*>(src);
        std::copy_n(src_ptr, count, dst_ptr);
    }
}

template <typename T>
void BuiltinTypeBase<T>::destroy_range(void* data, size_t count) const
{
    if constexpr (!is_trivially_destructible_v<T>) {
        std::destroy_n(static_cast<T*>(data), count);
    }
}

template <typename T>
std::unique_ptr<const ConversionTrait>
BuiltinTypeBase<T>::convert_to(const Type& type) const noexcept
{
    static const auto conversion_table = make_conversion_to_table<T>();

    if (&type == this || type.type_info() == type_info()) {
        return std::make_unique<ConversionTraitImpl<T, T>>(this, this);
    }

    Hash<string_view> hasher;
    if (const auto it = conversion_table.find(hasher(type.id()));
        it != conversion_table.end()) {
        return it->second->make(&type, this);
    }

    return Type::convert_to(type);
}

template <typename T>
std::unique_ptr<const ConversionTrait>
BuiltinTypeBase<T>::convert_from(const Type& type) const noexcept
{
    static const auto conversion_table = make_conversion_from_table<T>();

    if (&type == this || type.type_info() == type_info()) {
        return std::make_unique<ConversionTraitImpl<T, T>>(this, this);
    }

    Hash<string_view> hasher;
    if (const auto it = conversion_table.find(hasher(type.id()));
        it != conversion_table.end()) {
        return it->second->make(this, &type);
    }

    return Type::convert_from(type);
}

template <typename T>
const BuiltinTrait*
BuiltinTypeBase<T>::get_builtin_trait(BuiltinTraitID id) const noexcept
{
    switch (id) {
        case BuiltinTraitID::Comparison: return &m_comparison_trait;
        case BuiltinTraitID::Arithmetic: return &m_arithmetic_trait;
        case BuiltinTraitID::Number: return &m_number_trait;
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}

// template <typename T>
// const Trait* BuiltinTypeBase<T>::get_trait(string_view id) const noexcept
// {
//     if (auto it = m_traits.find(id); it != m_traits.end()) {
//         return it->second.get();
//     }
//     return nullptr;
// }
//

template <typename T>
const std::ostream&
BuiltinTypeBase<T>::display(std::ostream& os, const void* value) const
{
    // Wrapping code should handle the case when value == nullptr,
    // so a null value jere is a bug
    RPY_DBG_ASSERT_NE(value, nullptr);
    return os << *static_cast<const T*>(value);
}

template <typename T>
hash_t BuiltinTypeBase<T>::hash_of(const void* value) const noexcept
{
    Hash<T> hasher;
    if (RPY_UNLIKELY(value == nullptr)) { return hasher(0.0); }
    return hasher(*static_cast<const T*>(value));
}

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_METHODS_H
