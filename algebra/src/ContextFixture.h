//
// Created by sam on 13/03/23.
//

#ifndef ROUGHPY_CONTEXTFIXTURE_H
#define ROUGHPY_CONTEXTFIXTURE_H

#include <gtest/gtest.h>

#include <roughpy/config/implementation_types.h>
#include <roughpy/scalars/scalar_type.h>
#include "context.h"

namespace rpy {
namespace algebra {
namespace testing {

class ContextFixture : public ::testing::Test {
    static constexpr deg_t width = 5;
    static constexpr deg_t depth = 5;

    const scalars::ScalarType* stype;

    context_pointer ctx;

public:
    ContextFixture();


};

}// namespace testing
}// namespace algebra
}// namespace rpy

#endif//ROUGHPY_CONTEXTFIXTURE_H
