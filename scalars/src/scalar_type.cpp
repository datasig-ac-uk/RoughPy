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

#include <roughpy/scalars/scalar_type.h>

#include "scalar/casts.h"
#include <roughpy/scalars/scalar_array.h>

using namespace rpy;
using namespace rpy::scalars;

ScalarType::ScalarType(
        std::string name,
        std::string id,
        rpy::dimn_t alignment,
        devices::Device device,
        devices::TypeInfo type_info,
        rpy::scalars::RingCharacteristics characteristics
)
    : m_name(std::move(name)),
      m_id(std::move(id)),
      m_alignment(alignment),
      m_device(std::move(device)),
      m_info(type_info),
      m_characteristics(characteristics)
{}

ScalarArray ScalarType::allocate(dimn_t count) const
{
    return ScalarArray(this, m_device->raw_alloc(count, m_alignment));
}

void* ScalarType::allocate_single() const
{
    RPY_THROW(
            std::runtime_error,
            "single scalar allocation is not available "
            "for " + m_name
    );
    return nullptr;
}

void ScalarType::free_single(void* ptr) const
{
    RPY_THROW(
            std::runtime_error,
            "single scalar allocation is not available for " + m_name
    );
}

void ScalarType::convert_copy(
        rpy::scalars::ScalarArray& dst,
        const rpy::scalars::ScalarArray& src
) const
{
    if (dst.size() < src.size()) {
        RPY_THROW(std::runtime_error, "insufficient size for copy");
    }
    if (dst.device() != m_device) {
        RPY_THROW(std::runtime_error, "unable to copy into device memory");
    }
    if (src.device() != m_device) {
        RPY_THROW(std::runtime_error, "unable to copy from device memory");
    }

    if (!dtl::scalar_convert_copy(
                dst.mut_pointer(),
                dst.type_info(),
                src.pointer(),
                src.type_info()
        )) {
        RPY_THROW(std::runtime_error, "convert copy failed");
    }
}
