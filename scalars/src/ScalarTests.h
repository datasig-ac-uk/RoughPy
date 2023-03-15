//
// Created by sam on 15/03/23.
//

#ifndef ROUGHPY_SCALARTESTS_H
#define ROUGHPY_SCALARTESTS_H

#include <gtest/gtest.h>

#include "scalar_type.h"

namespace rpy {
namespace scalars {
namespace testing {

class ScalarTests : public ::testing::Test {
public:

    const scalars::ScalarType* dtype;
    const scalars::ScalarType* ftype;

    ScalarTests() : dtype(ScalarType::of<double>()), ftype(ScalarType::of<float>())
    {}

};

}// namespace testing
}// namespace scalars
}// namespace rpy

#endif//ROUGHPY_SCALARTESTS_H
