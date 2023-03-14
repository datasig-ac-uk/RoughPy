//
// Created by user on 13/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_TESTING_MOCK_CONTEXT_H
#define ROUGHPY_ALGEBRA_SRC_TESTING_MOCK_CONTEXT_H

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <roughpy/algebra/context.h>
#include <roughpy/scalars/scalar_type.h>

namespace rpy {
namespace algebra {
namespace testing {

class MockContext : public Context {

public:

    MockContext() : Context(2, 2, scalars::ScalarType::of<float>(), "mock")
    {}


};

}// namespace testing
}// namespace algebra
}// namespace rpy

#endif//ROUGHPY_ALGEBRA_SRC_TESTING_MOCK_CONTEXT_H
