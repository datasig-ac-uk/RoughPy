//
// Created by sam on 16/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_METHODS_H
#define ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_METHODS_H

#include "builtin_type.h"

#include <algorithm>


#include "roughpy/core/alloc.h"
#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/hash.h"
#include "roughpy/core/smart_ptr.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "conversion_factory.h"



namespace rpy::generics {


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
void* BuiltinTypeBase<T>::allocate_object() const
{
    // TODO: replace with small object allocator
    return aligned_alloc(alignof(T), alignof(T));
}

template <typename T>
void BuiltinTypeBase<T>::free_object(void* ptr) const
{
    // TODO: replace with small object allocator
    aligned_free(ptr);
}

template <typename T>
void BuiltinTypeBase<T>::copy_or_move(
        void* dst,
        const void* src,
        size_t count,
        bool RPY_UNUSED_VAR(move) // Move semantics makes no difference for T
) const noexcept
{

    if (RPY_UNLIKELY(dst == nullptr || count == 0)) {
        return;
    }

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
const BuiltinTrait* BuiltinTypeBase<T>::get_builtin_trait(BuiltinTraitID id
) const noexcept
{
    switch (id) {
        case BuiltinTraitID::Comparison:
            return &m_arithmetic_trait;
        case BuiltinTraitID::Hash:
            return &m_hash_trait;
        case BuiltinTraitID::Arithmetic:
            return &m_arithmetic_trait;
        case BuiltinTraitID::Number:
            return &m_number_trait;
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


}


#endif //ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_METHODS_H
