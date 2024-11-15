//
// Created by sammorley on 15/11/24.
//




#include <gtest/gtest.h>

#include "check.h"


TEST(CheckMacros, TestCheckFailStandard)
{
    EXPECT_THROW(RPY_CHECK(false), std::runtime_error);
}

TEST(CheckMacros, TestCHeckWithSpecifiedException)
{
    EXPECT_THROW(
        RPY_CHECK(false, "message", std::invalid_argument),
        std::invalid_argument
    );
}

TEST(CheckMacros, TestCheckEqFails)
{
    EXPECT_THROW(RPY_CHECK_EQ(1, 2), std::runtime_error);
}

TEST(CheckMacros, TestCheckEqFailsCustomException)
{
    EXPECT_THROW(
        RPY_CHECK_EQ(1, 2, std::invalid_argument),
        std::invalid_argument
    );
}

TEST(CheckMacros, TestCheckNeFails)
{
    EXPECT_THROW(RPY_CHECK_NE(1, 1), std::runtime_error);
}

TEST(CheckMacros, TestCheckNeFailsCustomException)
{
    EXPECT_THROW(
            RPY_CHECK_NE(1, 1, std::invalid_argument),
            std::invalid_argument
    );
}

TEST(CheckMacros, TestCheckLtFails)
{
    EXPECT_THROW(RPY_CHECK_LT(2, 1), std::runtime_error);
}

TEST(CheckMacros, TestCheckLtFailsCustomException)
{
    EXPECT_THROW(
            RPY_CHECK_LT(2, 1, std::invalid_argument),
            std::invalid_argument
    );
}

TEST(CheckMacros, TestCheckLeFails)
{
    EXPECT_THROW(RPY_CHECK_LE(2, 1), std::runtime_error);
}

TEST(CheckMacros, TestCheckLeFailsCustomException)
{
    EXPECT_THROW(
            RPY_CHECK_LE(2, 1, std::invalid_argument),
            std::invalid_argument
    );
}

TEST(CheckMacros, TestCheckGtFails)
{
    EXPECT_THROW(RPY_CHECK_GT(1, 2), std::runtime_error);
}

TEST(CheckMacros, TestCheckGtFailsCustomException)
{
    EXPECT_THROW(
            RPY_CHECK_GT(1, 2, std::invalid_argument),
            std::invalid_argument
    );
}

TEST(CheckMacros, TestCheckGeFails)
{
    EXPECT_THROW(RPY_CHECK_GE(1, 2), std::runtime_error);
}

TEST(CheckMacros, TestCheckGeFailsCustomException)
{
    EXPECT_THROW(
            RPY_CHECK_GE(1, 2, std::invalid_argument),
            std::invalid_argument
    );
}