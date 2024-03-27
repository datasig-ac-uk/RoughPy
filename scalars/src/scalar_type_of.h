//
// Created by sam on 3/27/24.
//


#ifndef SCALAR_TYPE_OF_H
#define SCALAR_TYPE_OF_H

#include "scalar_type.h"

namespace rpy { namespace scalars {

RPY_LOCAL
void register_scalar_type(const ScalarType* tp);
RPY_LOCAL
void unregister_scalar_type(string_view id);

}}


#endif //SCALAR_TYPE_OF_H
