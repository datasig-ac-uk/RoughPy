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

#include <roughpy/core/helpers.h>
#include "devices/kernel.h"

#include "devices/device_handle.h"

using namespace rpy;
using namespace rpy::devices;

KernelLaunchParams::KernelLaunchParams()
    : m_work_dims(0, 0, 0),
      m_group_size(0, 0, 0),
      m_offsets()
{}

bool KernelLaunchParams::has_work() const noexcept { return m_work_dims.x > 0; }

Size3 KernelLaunchParams::total_work_dims() const noexcept
{
    return m_work_dims;
}

Dim3 KernelLaunchParams::work_groups() const noexcept
{
    return m_group_size;
}

dimn_t KernelLaunchParams::total_work_size() const noexcept
{
    dimn_t result = m_work_dims.x;
    if (m_work_dims.y > 0) { result *= m_work_dims.y; }
    if (m_work_dims.z > 0) { result *= m_work_dims.z; }
    return result;
}

dsize_t KernelLaunchParams::num_dims() const noexcept
{
    if (m_work_dims.z > 0) {
        RPY_DBG_ASSERT(m_work_dims.x > 0 && m_work_dims.y > 0);
        return 3;
    }
    if (m_work_dims.y > 0) {
        RPY_DBG_ASSERT(m_work_dims.x > 0);
        return 2;
    }
    return 1;
}

Dim3 KernelLaunchParams::num_work_groups() const noexcept
{
    Dim3 result;
    if (m_group_size.x > 0) {
        result.x = round_up_divide(m_work_dims.x, m_group_size.x);
        if (m_group_size.y > 0) {
            result.y = round_up_divide(m_work_dims.y, m_group_size.y);
            if (m_group_size.z > 0) {
                result.z = round_up_divide(m_work_dims.z, m_group_size.z);
            }
        }
    }
    return result;
}

Size3 KernelLaunchParams::underflow_of_groups() const noexcept
{
    Size3 result;
    if (m_group_size.x > 0) {
        result.x = static_cast<dimn_t>(
                m_group_size.x - (m_work_dims.x % m_group_size.x)
        );

        if (m_group_size.y > 0) {
            result.y = static_cast<dimn_t>(
                    m_group_size.y - (m_work_dims.y % m_group_size.y)
            );

            if (m_group_size.z > 0) {
                result.z = static_cast<dimn_t>(
                        m_group_size.z - (m_work_dims.z % m_group_size.z)
                );
            }
        }
    }
    return result;
}
