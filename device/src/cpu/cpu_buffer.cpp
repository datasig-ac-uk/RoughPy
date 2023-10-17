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
// Created by user on 16/10/23.
//

#include "cpu_buffer.h"

#include <roughpy/device/device_handle.h>
#include <roughpy/device/device_object_base.h>

#include "opencl/ocl_device.h"
#include "opencl/ocl_handle_errors.h"

#include "cpu_device.h"

using namespace rpy;
using namespace rpy::device;

CPUBuffer::CPUBuffer(cl_mem buffer, CPUDevice dev)
    : ocl_buffer(buffer, dev->ocl_device()), is_ocl(true)
{}

CPUBuffer::CPUBuffer(void* raw_ptr, dimn_t size)
    : raw_buffer{ raw_ptr, size, IsOwned}, is_ocl(false)
{}

CPUBuffer::CPUBuffer(const void* raw_ptr, dimn_t size)
    : raw_buffer { const_cast<void*>(raw_ptr), size, IsConst }, is_ocl(false)
{}

CPUBuffer::~CPUBuffer() {
    if (is_ocl) {
        ocl_buffer.~OCLBuffer();
    }

    if (raw_buffer.flags & IsOwned) {

    }

}

BufferMode CPUBuffer::mode() const
{
    if (is_ocl) {
        return ocl_buffer.mode();
    }

    if (raw_buffer.flags & IsConst) {
        return BufferMode::Read;
    }
    return BufferMode::ReadWrite;
}
dimn_t CPUBuffer::size() const {
    if (is_ocl) {
        return ocl_buffer.size();
    }
    return raw_buffer.size;
}
void* CPUBuffer::ptr() {
    if (is_ocl) {
        return ocl_buffer.ptr();
    }
    return raw_buffer.ptr;
}
std::unique_ptr<rpy::device::dtl::InterfaceBase> CPUBuffer::clone() const
{
    return nullptr;
}
Device CPUBuffer::device() const noexcept {
    return CPUDeviceHandle::get();
}
