//
// Created by sam on 16/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_H
#define ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_H

#include <iosfwd>
#include <typeinfo>

#include <roughpy/core/macros.h>

#include "roughpy/generics/type.h"

#include "roughpy/generics/arithmetic_trait.h"
#include "roughpy/generics/builtin_trait.h"
#include "roughpy/generics/comparison_trait.h"
#include "roughpy/generics/conversion_trait.h"
#include "roughpy/generics/number_trait.h"

namespace rpy::generics {

template <typename T>
class BuiltinTypeBase : public Type
{
    ArithmeticTraitImpl<T> m_arithmetic_trait;
    ComparisonTraitImpl<T> m_comparison_trait;
    NumberTraitImpl<T> m_number_trait;


    hash_t hash_with_type(const Type& other_type) const noexcept
    {
        Hash<string_view> hasher;
        hash_t result = hasher(this->id());
        hash_combine(result, hasher(other_type.id()));
        return result;
    }

protected:
    BuiltinTypeBase()
        : m_arithmetic_trait(this, this),
          m_comparison_trait(this),
          m_number_trait(this, this)
    {}

    void inc_ref() const noexcept override;
    bool dec_ref() const noexcept override;
    void* allocate_object() const override;
    void free_object(void*) const override;

public:
    RPY_NO_DISCARD intptr_t ref_count() const noexcept override { return 1; }
    RPY_NO_DISCARD string_view id() const noexcept override;

    RPY_NO_DISCARD BasicProperties
    basic_properties() const noexcept override
    {
        return basic_properties_of<T>();
    }
    RPY_NO_DISCARD size_t object_size() const noexcept override
    {
        return sizeof(T);
    }

    RPY_NO_DISCARD const std::type_info& type_info() const noexcept override;

    void copy_or_fill(
            void* dst,
            const void* src,
            size_t count,
            bool uninit
    ) const noexcept override;
    void destroy_range(void* data, size_t count) const override;


    bool parse_from_string(void* data, string_view str) const noexcept override;

    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait>
    convert_to(const Type& type) const noexcept override;
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait>
    convert_from(const Type& type) const noexcept override;
    RPY_NO_DISCARD const BuiltinTrait*
    get_builtin_trait(BuiltinTraitID id) const noexcept override;
    // RPY_NO_DISCARD const Trait* get_trait(string_view id
    // ) const noexcept override;
    const std::ostream&
    display(std::ostream& os, const void* value) const override;

    hash_t hash_of(const void* value) const noexcept override;
};

}// namespace rpy::generics

#endif// ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_H
