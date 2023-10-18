//
// Created by sam on 18/10/23.
//


#include <gtest/gtest.h>

#include <roughpy/device/device_handle.h>


using namespace rpy;


TEST(CPUDevice, TestDeviceInfo) {
    auto device = devices::get_cpu_device();

    auto info = device->info();

    ASSERT_EQ(info.device_type, devices::DeviceType::CPU);
    ASSERT_EQ(info.device_id, 0);
}


TEST(CPUDevice, TestSupportedBasicTypes) {
    auto device = devices::get_cpu_device();

    devices::TypeInfo f32_info{
            devices::TypeCode::Float,
            32,
            1
    };
    EXPECT_TRUE(device->supports_type(f32_info));

    devices::TypeInfo f64_info{
            devices::TypeCode::Float,
            64,
            1
    };
    EXPECT_TRUE(device->supports_type(f64_info));
}

TEST(CPUDevice, TestDefaultQueueIsDefault) {
    auto device = devices::get_cpu_device();

    ASSERT_TRUE(device->get_default_queue().is_default());
}


TEST(CPUDevice, TestAllocBufferCorrectSize) {
    auto device = devices::get_cpu_device();

    auto buffer = device->raw_alloc(128, 16);

    EXPECT_EQ(buffer.size(), 128);
}
