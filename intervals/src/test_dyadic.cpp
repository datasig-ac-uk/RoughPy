//
// Created by user on 16/03/23.
//

#include <gtest/gtest.h>

#include "dyadic.h"

using namespace rpy::intervals;

TEST(Dyadictests, test_rebase_dyadic_1) {
    Dyadic val{1, 0};
    val.rebase(1);
    ASSERT_TRUE(dyadic_equals(val, Dyadic{2, 1}));
}

TEST(Dyadictests, test_rebase_dyadic_5) {
    Dyadic val{1, 0};
    val.rebase(5);
    ASSERT_TRUE(dyadic_equals(val, Dyadic{32, 5}));
}
