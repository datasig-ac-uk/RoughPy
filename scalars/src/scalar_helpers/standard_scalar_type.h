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
// Created by user on 03/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_SCALAR_HELPERS_STANDARD_SCALAR_TYPE_H_
#define ROUGHPY_SCALARS_SRC_SCALAR_HELPERS_STANDARD_SCALAR_TYPE_H_

#include "scalar_type.h"
#include "scalar_array.h"
#include "scalar.h"
#include "random/standard_random_generator.h"

#include <roughpy/platform/devices/device_handle.h>
#include <roughpy/platform/devices/host_device.h>

#include "scalar/casts.h"

#include <algorithm>
#include <unordered_set>

namespace rpy {
namespace scalars {
namespace dtl {

template <typename ScalarImpl>
class StandardScalarType : public ScalarType
{
    mutable std::unordered_set<void*> m_allocated;

    static std::unique_ptr<RandomGenerator> get_mt19937_generator(
        const ScalarType* tp,
        Slice<seed_int_t> seed);
    static std::unique_ptr<RandomGenerator> get_pcg_generator(
        const ScalarType* tp,
        Slice<seed_int_t> seed);

protected:
    StandardScalarType(string name, string id, RingCharacteristics chars)
        : ScalarType(std::move(name),
                     std::move(id),
                     alignof(ScalarImpl),
                     devices::get_host_device(),
                     devices::type_info<ScalarImpl>(),
                     chars
        )
    {
        register_rng_getter("mt19937", &get_mt19937_generator);
        register_rng_getter("pcg", &get_pcg_generator);
    }

public:
    ScalarArray allocate(dimn_t count) const override;
    void* allocate_single() const override;
    void free_single(void* ptr) const override;
    void convert_copy(ScalarArray& dst, const ScalarArray& src) const override;
    void assign(ScalarArray& dst, Scalar value) const override;

};

template <typename ScalarImpl>
ScalarArray StandardScalarType<ScalarImpl>::allocate(dimn_t count) const
{
    RPY_CHECK(count > 0);
    ScalarArray result(this, m_device->raw_alloc(count*m_info.bytes, m_info.alignment));;
    {
        auto slice = result.mut_buffer().template as_mut_slice<ScalarImpl>();
        std::uninitialized_fill(slice.begin(), slice.end(), ScalarImpl(0));
    }

    return result;
}

template <typename ScalarImpl>
void* StandardScalarType<ScalarImpl>::allocate_single() const
{
    guard_type access(m_lock);
    auto [pos, inserted] = m_allocated.insert(
        static_cast<void*>(new ScalarImpl()));
    RPY_DBG_ASSERT(inserted);
    return *pos;
}

template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::free_single(void* ptr) const
{
    guard_type access(m_lock);
    auto found = m_allocated.find(ptr);
    if (found == m_allocated.end()) {
        RPY_THROW(std::runtime_error,
                  "Attempting to free scalar allocated "
                  "with as a different type");
    }
    delete static_cast<ScalarImpl*>(*found);
    m_allocated.erase(found);
}

template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::convert_copy(
    ScalarArray& dst,
    const ScalarArray& src
) const
{
    if (src.empty()) {
        // Nothing to do.
        return;
    }
    if (dst.is_const()) {
        RPY_THROW(std::runtime_error, "unable to copy into const array");
    }

    auto dst_type = dst.type();
    auto src_type = src.type();
    if (dst_type) {
        if (dst_type != this) {
            (*dst_type)->convert_copy(dst, src);
            return;
        }

        if (src_type && (*src_type)->device() != m_device) {
            (*src_type)->convert_copy(dst, src);
            return;
        }
    } else if (src_type && src_type != this) {
        (*src_type)->convert_copy(dst, src);
    }

    auto src_info = src.type_info();
    auto dst_info = dst.type_info();
    /*
     * Now one of the following cases must be true:
     *  1) dst_type is defined and is equal to this and src has the same
     *     devices as this.
     *  2) src_type is defined and is equal to this.
     *  3) neither src_type nor dst_type are defined.
     */
    auto src_size = src.size();
    auto dst_cap = dst.capacity();
    if (dst.is_null()) {
        // If the dst array is empty, allocate enough space for the new data.
        dst = allocate(src_size);
    } else if (dst_cap < src_size * dst_info.bytes) {
        if (dst.is_owning()) {
            if (dst_type) {
                // If dst_type is defined, then it is this so a simple
                // reallocation can be used.
                dst = allocate(src_size);
            } else {
                // dst type is not defined, so is a trivial type fully
                // described by the TypeInfo struct.
                auto new_buf = m_device->raw_alloc(src_size * dst_info.bytes,
                                                   dst_info.alignment);
                dst = ScalarArray(dst_info, std::move(new_buf));
            }
        } else {
            RPY_THROW(std::runtime_error, "cannot resize a borrowed array");
        }
    } else if (dst.is_owning() && dst_cap > src_size) {
        // Just a little care is needed here to make sure the size of the
        // final array is correct.
        if (dst_type) {
            dst = ScalarArray(*dst_type, std::move(dst.mut_buffer()));
        } else { dst = ScalarArray(dst_info, std::move(dst.mut_buffer())); }
    }

    /*
     * Right, now the dst buffer should be large enough to accept the data
     * from src_buffer. There shouldn't be anything else to do that is
     * specific to this type, so pass down to the generic copy convert function.
     */

    ScalarType::convert_copy(dst, src);
}

template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::assign(ScalarArray& dst,
                                            Scalar value)
const
{
    if (dst.is_null()) {
        RPY_THROW(std::invalid_argument, "destination array is not valid");
    }
    auto dst_type = dst.type();
    if (dst_type && *dst_type != this) {
        (*dst_type)->assign(dst, std::move(value));
        return;
    }
    if (!dst_type && dst.type_info() != m_info) {
        RPY_THROW(std::invalid_argument, "dst has incorrect type");
    }

    {
        auto slice = dst.template as_mut_slice<ScalarImpl>();
        std::fill(slice.begin(), slice.end(), scalar_cast<ScalarImpl>(value));
    }

}


template <typename ScalarImpl>
std::unique_ptr<RandomGenerator> StandardScalarType<ScalarImpl>::
get_mt19937_generator(const ScalarType* tp, Slice<seed_int_t> seed)
{
    return std::make_unique<StandardRandomGenerator<
        ScalarImpl, std::mt19937_64>>(tp, seed);
}

template <typename ScalarImpl>
std::unique_ptr<RandomGenerator> StandardScalarType<ScalarImpl>::
get_pcg_generator(const ScalarType* tp, Slice<seed_int_t> seed)
{
    return std::make_unique<StandardRandomGenerator<
        ScalarImpl, pcg64>>(tp, seed);
}


}// namespace dtl
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_SCALAR_HELPERS_STANDARD_SCALAR_TYPE_H_
