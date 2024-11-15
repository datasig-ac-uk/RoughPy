//
// Created by sammorley on 15/11/24.
//




#include <gtest/gtest.h>

#include "check.h"




TEST(CheckMacros, TestCheckEqFails)
{
    EXPECT_THROW(RPY_CHECK_EQ(1, 2), std::runtime_error);
}