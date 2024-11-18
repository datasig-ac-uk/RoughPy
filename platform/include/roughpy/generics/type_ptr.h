//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_GENERICS_TYPE_PTR_H
#define ROUGHPY_GENERICS_TYPE_PTR_H

#include <roughpy/core/smart_ptr.h>

namespace rpy::generics {

class Type;

using TypePtr = Rc<const Type>;



}

#endif //ROUGHPY_GENERICS_TYPE_PTR_H
