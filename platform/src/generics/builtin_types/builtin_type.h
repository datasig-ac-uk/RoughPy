//
// Created by sam on 16/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_H
#define ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_H


#include <typeinfo>
#include <unordered_map>

#include <roughpy/core/macros.h>

#include "generics/type.h"

#include "generics/arithmetic_trait.h"
#include "generics/builtin_trait.h"
#include "generics/comparison_trait.h"
#include "generics/conversion_trait.h"
#include "generics/number_trait.h"

namespace rpy::generics {






template <typename T>
class ROUGHPY_PLATFORM_NO_EXPORT BuiltinTypeBase : public Type
{
    ArithmeticTraitImpl<T> m_arithmetic_trait;
    ComparisonTraitImpl<T> m_comparison_trait;
    NumberTraitImpl<T> m_number_trait;


    // std::unordered_map<string_view, std::unique_ptr<const Trait>> m_traits;

    hash_t hash_with_type(const Type& other_type) const noexcept
    {
        Hash<string_view> hasher;
        hash_t result = hasher(this->id());
        hash_combine(result, hasher(other_type.id()));
        return result;
    }

protected:

    BuiltinTypeBase()
        : Type(&typeid(T), sizeof(T), basic_properties_of<T>()),
          m_arithmetic_trait(this, this),
          m_comparison_trait(this),
          m_number_trait(this, this)
    {}

    void inc_ref() const noexcept override;
    bool dec_ref() const noexcept override;
    void* allocate_object() const override;
    void free_object(void*) const override;



public:
    void copy_or_move(void* dst, const void* src, size_t count, bool move)
            const noexcept override;
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait>
    convert_to(const Type& type) const noexcept override;
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait>
    convert_from(const Type& type) const noexcept override;
    RPY_NO_DISCARD const BuiltinTrait* get_builtin_trait(BuiltinTraitID id
    ) const noexcept override;
    // RPY_NO_DISCARD const Trait* get_trait(string_view id
    // ) const noexcept override;
    const std::ostream&
    display(std::ostream& os, const void* value) const override;


};


}// namespace rpy::generics

#endif //ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_H
