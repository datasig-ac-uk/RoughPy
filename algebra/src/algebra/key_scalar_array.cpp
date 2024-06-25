//
// Created by sam on 3/28/24.
//

#include "key_scalar_array.h"

using namespace rpy;
using namespace rpy::algebra;

KeyScalarArray::KeyScalarArray(scalars::TypePtr type, dimn_t size)
    : ScalarArray(type, size)
{}
KeyScalarArray::~KeyScalarArray() = default;
