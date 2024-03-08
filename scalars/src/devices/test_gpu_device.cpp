//
// Created by sam on 20/10/23.
//


#include <gtest/gtest.h>
#include <roughpy/platform/devices/core.h>

#include <roughpy/platform/devices/device_handle.h>

using namespace rpy;

namespace {

class GPUDeviceTests : public ::testing::Test {
protected:
    devices::Device device;

    virtual void SetUp() {
        devices::DeviceSpecification spec {
            devices::DeviceCategory::GPU,
            0
        };
        auto dv = devices::get_device(spec);
        if (dv) {
            device = *dv;
        } else {
            GTEST_SKIP() << "No GPU device available";
        }

    }
};

}


TEST_F(GPUDeviceTests, TestDeviceCategory) {
    ASSERT_EQ(device->category(), devices::DeviceCategory::GPU);
}
