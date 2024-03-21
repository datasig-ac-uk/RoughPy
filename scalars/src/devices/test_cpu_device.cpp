// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by sam on 18/10/23.
//

#include <gtest/gtest.h>

#include <roughpy/platform/devices/host_device.h>
#include <roughpy/platform/devices/buffer.h>
#include <roughpy/platform/devices/event.h>
#include <roughpy/platform/devices/kernel.h>
#include <roughpy/platform/devices/queue.h>

using namespace rpy;

TEST(CPUDevice, TestDeviceInfo)
{
    auto device = devices::get_host_device();

    auto info = device->info();

    ASSERT_EQ(info.device_type, devices::DeviceType::CPU);
    ASSERT_EQ(info.device_id, 0);
}

TEST(CPUDevice, TestSupportedBasicTypes)
{
    auto device = devices::get_host_device();

    devices::TypeInfo f32_info{devices::TypeCode::Float, 32, 1};
    EXPECT_TRUE(device->supports_type(f32_info));

    devices::TypeInfo f64_info{devices::TypeCode::Float, 64, 1};
    EXPECT_TRUE(device->supports_type(f64_info));
}

TEST(CPUDevice, TestDefaultQueueIsDefault)
{
    auto device = devices::get_host_device();

    ASSERT_TRUE(device->get_default_queue().is_default());
}

TEST(CPUDevice, TestAllocBufferCorrectSize)
{
    auto device = devices::get_host_device();

    auto buffer = device->raw_alloc(128, 16);

    EXPECT_EQ(buffer.size(), 128);
}

namespace {

class TestCPUDeviceOCL : public ::testing::Test
{

protected:
    devices::Device device;

    void SetUp() override
    {
        device = devices::get_host_device();
        if (!device->has_compiler()) {
            GTEST_SKIP() << "No OpenCL device for CPU";
        }
    }
};

}// namespace

TEST_F(TestCPUDeviceOCL, TestCreateKernelFromSource)
{
    devices::ExtensionSourceAndOptions ext;

    ext.sources.push_back(R"cl(
        __kernel void test_kernel(__global float* x) {
            size_t id = get_global_id(0);
            x[id] *= 2.0;
        }
    )cl");

    auto kernel = device->compile_kernel_from_str(ext);
    auto k_device = kernel->device();

    EXPECT_TRUE(static_cast<bool>(kernel));
    EXPECT_EQ(kernel->name(), "test_kernel");

    dimn_t count = 50;
    auto buf = device->raw_alloc(count * sizeof(float), sizeof(float));
    auto* raw = static_cast<float*>(buf.ptr());

    for (dimn_t i = 0; i < count; ++i) { raw[i] = static_cast<float>(i); }

    devices::Buffer kbuf;
    buf.to_device(kbuf, k_device);
    devices::KernelLaunchParams params(devices::Size3{count}, devices::Dim3{1});

    (*kernel)(params, kbuf);

    kbuf.to_device(buf, device);
    auto result = buf.as_slice<float>();
    ASSERT_EQ(result.size(), count);
    for (dimn_t i = 0; i < count; ++i) {
        ASSERT_EQ(result[i], 2*static_cast<float>(i))
                << "Index " << i << " mismatch";
    }
}
