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
using namespace rpy::devices;

CPUBuffer::CPUBuffer(CPUBuffer::RawBuffer raw, CPUBuffer::Flags arg_flags)
    : raw_buffer(std::move(raw)),
      flags(arg_flags)
{
    RPY_DBG_ASSERT(raw_ref_count(std::memory_order_acq_rel) > 0);
    inc_ref();
}

CPUBuffer::CPUBuffer(void* raw_ptr, dimn_t size, atomic_t rc)
    : raw_buffer{raw_ptr, size, rc},
      flags()
{}

CPUBuffer::CPUBuffer(const void* raw_ptr, dimn_t size, atomic_t rc)
    : raw_buffer{const_cast<void*>(raw_ptr), size, rc},
      flags(IsConst)
{}

CPUBuffer::~CPUBuffer()
{
    if (dec_ref() == 1) {
        RPY_DBG_ASSERT(raw_ref_count(std::memory_order_acq_rel) == 0);
        CPUDeviceHandle::get()->raw_free(raw_buffer.ptr, raw_buffer.size);
        raw_buffer.ptr = nullptr;
        raw_buffer.size = 0;
    }
}

BufferMode CPUBuffer::mode() const
{
    if (flags & IsConst) { return BufferMode::Read; }
    return BufferMode::ReadWrite;
}
dimn_t CPUBuffer::size() const
{
    return raw_buffer.size;
}
void* CPUBuffer::ptr()
{
    return raw_buffer.ptr;
}
std::unique_ptr<rpy::devices::dtl::InterfaceBase> CPUBuffer::clone() const {
    return std::unique_ptr<CPUBuffer>(new CPUBuffer(raw_buffer, flags));
}
Device CPUBuffer::device() const noexcept { return CPUDeviceHandle::get(); }
dimn_t CPUBuffer::ref_count() const noexcept
{
    return raw_ref_count();
}
