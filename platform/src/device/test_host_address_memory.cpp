//
// Created by sammorley on 22/11/24.
//

#include <gtest/gtest.h>

#include <roughpy/device/device_handle.h>
#include <roughpy/device/host_address_memory.h>
#include <roughpy/generics/type.h>

using namespace rpy;
using namespace rpy::device;

TEST(HostAddressMemory, TestReadOnlyAccess)
{
    double value = 3.14159265358979323846;
    HostAddressMemory mem(
            *generics::Type::of<double>(),
            *DeviceHandle::host(),
            &value,
            1,
            sizeof(double),
            MemoryMode::ReadOnly
    );


    EXPECT_THROW(mem.data(), std::runtime_error);
}