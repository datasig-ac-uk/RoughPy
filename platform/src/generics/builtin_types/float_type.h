//
// Created by sammorley on 18/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_FLOAT_TYPE_H
#define ROUGHPY_GENERICS_INTERNAL_FLOAT_TYPE_H

#include "builtin_type.h"


namespace rpy::generics {

extern template class BuiltinTypeBase<float>;

class FloatType : public BuiltinTypeBase<float> {

public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;
    RPY_NO_DISCARD string_view id() const noexcept override;

    static const Type* get() noexcept;
};

} // rpy::generics

#endif //ROUGHPY_GENERICS_INTERNAL_FLOAT_TYPE_H
