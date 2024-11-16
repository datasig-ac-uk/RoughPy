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
#include "generics/hash_trait.h"
#include "generics/number_trait.h"

namespace rpy::generics {

template <typename T>
class BuiltinTypeBase : public Type
{
    ArithmeticTraitImpl<T> m_arithmetic_trait;
    ComparisonTraitImpl<T> m_comparison_trait;
    HashTraitImpl<T> m_hash_trait;
    NumberTrait m_number_trait; // Fix


    std::unordered_map<string_view, std::unique_ptr<const Trait>> m_traits;


protected:

    BuiltinTypeBase()
        : Type(&typeid(T), sizeof(T), basic_properties_of<T>())
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
    RPY_NO_DISCARD const Trait* get_trait(string_view id
    ) const noexcept override;
    const std::ostream&
    display(std::ostream& os, const void* value) const override;
};


}// namespace rpy::generics

#endif //ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_H
