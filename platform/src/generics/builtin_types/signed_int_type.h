//
// Created by sammorley on 18/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_SIGNED_INT_TYPE_H
#define ROUGHPY_GENERICS_INTERNAL_SIGNED_INT_TYPE_H

#include <roughpy/core/types.h>

#include "builtin_type.h"

namespace rpy::generics {


extern template class BuiltinTypeBase<int8_t>;
extern template class BuiltinTypeBase<int16_t>;
extern template class BuiltinTypeBase<int32_t>;
extern template class BuiltinTypeBase<int64_t>;



class SignedInt8Type : public BuiltinTypeBase<int8_t> {
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;

    static const Type* get() noexcept;
};

class SignedInt16Type : public BuiltinTypeBase<int16_t>
{
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;

    static const Type* get() noexcept;
};

class SignedInt32Type : public BuiltinTypeBase<int32_t>
{
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;

    static const Type* get() noexcept;
};

class SignedInt64Type : public BuiltinTypeBase<int64_t>
{
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override ;

    static const Type* get() noexcept;
};



} // rpy::generics

#endif //ROUGHPY_GENERICS_INTERNAL_SIGNED_INT_TYPE_H
