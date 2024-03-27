//
// Created by sam on 3/27/24.
//

#ifndef ROUGPY_ALGEBRA_KEY_SCALAR_ARRAY_H
#define ROUGPY_ALGEBRA_KEY_SCALAR_ARRAY_H

#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>

#include <roughpy/scalars/devices/core.h>
#include <roughpy/scalars/scalar_array.h>

#include "basis_key.h"
#include "key_array.h"


namespace rpy { namespace algebra {


class ROUGHPY_ALGEBRA_EXPORT KeyScalarArray : public scalars::ScalarArray
{
    KeyArray m_keys;




};


}}


#endif //ROUGPY_ALGEBRA_KEY_SCALAR_ARRAY_H
