//
// Created by sammorley on 18/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_UNSIGNED_INT_TYPE_H
#define ROUGHPY_GENERICS_INTERNAL_UNSIGNED_INT_TYPE_H


#include <roughpy/core/types.h>

#include "builtin_type.h"


namespace rpy::generics {

extern template class BuiltinTypeBase<uint8_t>;
extern template class BuiltinTypeBase<uint16_t>;
extern template class BuiltinTypeBase<uint32_t>;
extern template class BuiltinTypeBase<uint64_t>;

class UnsignedInt8Type : public BuiltinTypeBase<uint8_t> {
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;

    static const Type* get() noexcept;
};

class UnsignedInt16Type : public BuiltinTypeBase<uint16_t>
{
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;

    static const Type* get() noexcept;
};

class UnsignedInt32Type : public BuiltinTypeBase<uint32_t>
{
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;

    static const Type* get() noexcept;
};

class UnsignedInt64Type : public BuiltinTypeBase<uint64_t>
{
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;

    static const Type* get() noexcept;
};


} // rpy::generics

#endif //ROUGHPY_GENERICS_INTERNAL_UNSIGNED_INT_TYPE_H
