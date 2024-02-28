//
// Created by sam on 13/11/23.
//

#include "scalar_interface.h"

using namespace rpy;
using namespace rpy::scalars;

ScalarInterface::~ScalarInterface() = default;
void ScalarInterface::add_inplace(const Scalar& other) {}
void ScalarInterface::sub_inplace(const Scalar& other) {}
void ScalarInterface::mul_inplace(const Scalar& other) {}
void ScalarInterface::div_inplace(const Scalar& other) {}
