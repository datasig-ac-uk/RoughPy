//
// Created by sam on 16/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_DOUBLE_TYPE_H
#define ROUGHPY_GENERICS_INTERNAL_DOUBLE_TYPE_H

#include "builtin_type.h"

namespace rpy::generics {

extern template class BuiltinTypeBase<double>;

class DoubleType : public BuiltinTypeBase<double>
{
public:
    using BuiltinTypeBase::BuiltinTypeBase;

    RPY_NO_DISCARD string_view name() const noexcept override;
    RPY_NO_DISCARD string_view id() const noexcept override;

    static const Type* get() noexcept;
};


}


#endif //ROUGHPY_GENERICS_INTERNAL_DOUBLE_TYPE_H
