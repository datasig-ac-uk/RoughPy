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
// Created by user on 11/10/23.
//

#include "devices/device_object_base.h"

#include <memory>

#include "roughpy/core/types.h"

#include "devices/core.h"

using namespace rpy;
using namespace rpy::devices;

rpy::devices::dtl::InterfaceBase::~InterfaceBase() = default;

dimn_t devices::dtl::InterfaceBase::ref_count() const noexcept { return 1; }

DeviceType devices::dtl::InterfaceBase::type() const noexcept {
    return DeviceType::CPU;
}

std::unique_ptr<rpy::devices::dtl::InterfaceBase>
rpy::devices::dtl::InterfaceBase::clone() const
{
    return nullptr;
}

Device devices::dtl::InterfaceBase::device() const noexcept
{
    return Device(nullptr);
}

void* devices::dtl::InterfaceBase::ptr() noexcept { return nullptr; }

const void* devices::dtl::InterfaceBase::ptr() const noexcept
{
    return nullptr;
}

typename devices::dtl::InterfaceBase::reference_count_type
devices::dtl::InterfaceBase::inc_ref() noexcept
{
    return 0;
}
typename devices::dtl::InterfaceBase::reference_count_type
devices::dtl::InterfaceBase::dec_ref() noexcept
{
    return 0;
}
