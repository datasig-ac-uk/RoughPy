//
// Created by sam on 24/10/23.
//

#include <gtest/gtest.h>

#include <sstream>

#include "ocl_version.h"

using namespace rpy::devices;

std::ostream& rpy::devices::operator<<(
        std::ostream& os,
        const rpy::devices::OCLVersion& version
)
{
    return os << version.major() << '.' << version.minor() << '.'
              << version.patch();
}

TEST(TestOCLVersion, OCLParseVersionString11Extra)
{
    OCLVersion parsed("OpenCL 1.1 Mesa 22.3.6");
    OCLVersion expected{1, 1};

    EXPECT_EQ(parsed, expected);
}

TEST(TestOCLVersion, OCLParseVersionString11NoExtra)
{
    OCLVersion parsed("OpenCL 1.1");
    OCLVersion expected{1, 1};

    EXPECT_EQ(parsed, expected);
}

TEST(TestOCLVersion, OCLParseVersionString30Extra)
{
    OCLVersion parsed("OpenCL 3.0 (Build 0)");
    OCLVersion expected{3, 0};

    EXPECT_EQ(parsed, expected);
}

TEST(TestOCLVersion, StreamOutputOperator)
{

    OCLVersion version{2, 2};

    std::stringstream ss;
    ss << version;

    EXPECT_EQ(ss.str(), "2.2.0");
}
