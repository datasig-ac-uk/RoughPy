//
// Created by sammorley on 25/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_INTEGER_TYPE_H
#define ROUGHPY_GENERICS_INTERNAL_INTEGER_TYPE_H

#include "roughpy/generics/type.h"


namespace rpy {
namespace generics {

class IntegerType : public Type {
    IntegerType();
protected:
    void inc_ref() const noexcept override;
    RPY_NO_DISCARD bool dec_ref() const noexcept override;

public:
    RPY_NO_DISCARD intptr_t ref_count() const noexcept override;
    RPY_NO_DISCARD const std::type_info& type_info() const noexcept override;
    RPY_NO_DISCARD BasicProperties basic_properties() const noexcept override;
    RPY_NO_DISCARD size_t object_size() const noexcept override;
    RPY_NO_DISCARD string_view name() const noexcept override;
    RPY_NO_DISCARD string_view id() const noexcept override;

protected:
    void* allocate_object() const override;
    void free_object(void* ptr) const override;

public:
    bool parse_from_string(void* data, string_view str) const noexcept override;
    void copy_or_move(
            void* dst,
            const void* src,
            size_t count,
            bool move
    ) const override;
    void destroy_range(void* data, size_t count) const override;
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait>
    convert_to(const Type& type) const noexcept override;
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait>
    convert_from(const Type& type) const noexcept override;
    RPY_NO_DISCARD const BuiltinTrait*
    get_builtin_trait(BuiltinTraitID id) const noexcept override;
    const std::ostream&
    display(std::ostream& os, const void* value) const override;
    hash_t hash_of(const void* value) const noexcept override;

    static TypePtr get() noexcept;
};

} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_INTEGER_TYPE_H