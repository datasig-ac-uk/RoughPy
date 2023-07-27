// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_SCALARS_SCALAR_TYPE_HELPER_H_
#define ROUGHPY_SCALARS_SCALAR_TYPE_HELPER_H_

#include "scalar_pointer.h"
#include "scalar_type.h"
#include "scalars_fwd.h"

namespace rpy {
namespace scalars {

namespace impl_helpers {
/*!
 * @brief Functions to help implement your own scalar types
 */

template <typename S>
inline S try_convert(const ScalarPointer& other)
{
    if (other.is_null()) { return S(0); }

    const auto* stype = ScalarType::of<S>();
    const auto* type = other.type();
    if (type == stype) {
        // ScalarType::of<S>() is the CPU based version of the scalar,
        // so dereferencing is a safe operation
        return *other.template raw_cast<const S*>();
    }

    // The scalar type differs from stype, so we need to construct
    // the value properly. Call the convert-copy method on the type
    // to fill in the result
    S result;
    // throws if no conversion is possible.
    stype->convert_copy({stype, &result}, other, 1);
    return result;
}

template <typename S, typename T>
inline enable_if_t<is_integral<T>::value> copy_convert_basic(
        S* RPY_RESTRICT dptr, const T* RPY_RESTRICT sptr, dimn_t count
)
{
    for (dimn_t i = 0; i < count; ++i, ++dptr, ++sptr) {
        construct_inplace(dptr, *sptr);
    }
}

template <typename S>
inline void
copy_convert(ScalarPointer& dst, const ScalarPointer& src, dimn_t count)
{
    // dst type taken as reference type. If this is not the case,
    // the caller should instead call dst.type()->convert_copy(dst, src, count)
    const auto* type = dst.type();

    auto* dst_ptr = dst.template raw_cast<S*>();

    if (src.type() == nullptr) {
        switch (src.simple_integer_config()) {
            case flags::UnsignedInteger8:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const uint8_t*>(), count
                );
                break;
            case flags::UnsignedInteger16:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const uint16_t*>(), count
                );
                break;
            case flags::UnsignedInteger32:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const uint32_t*>(), count
                );
                break;
            case flags::UnsignedInteger64:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const uint64_t*>(), count
                );
                break;
            case flags::UnsignedSize:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const dimn_t*>(), count
                );
                break;
            case flags::SignedInteger8:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const int8_t*>(), count
                );
                break;
            case flags::SignedInteger16:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const int16_t*>(), count
                );
                break;
            case flags::SignedInteger32:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const int32_t*>(), count
                );
                break;
            case flags::SignedInteger64:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const int64_t*>(), count
                );
                break;
            case flags::SignedSize:
                copy_convert_basic(
                        dst_ptr, src.template raw_cast<const idimn_t*>(), count
                );
                break;
            default:
                RPY_THROW(
                        std::runtime_error,
                        "could not deduce the scalar "
                        "type of the source"
                );
        }
    } else {
        // If it isn't a simple type, look for a conversion function from the
        // table get_conversion throws if no suitable conversion is found.
        const auto& conversion = get_conversion(src.type()->id(), type->id());
        conversion(dst, src, count);
    }
}

template <typename S, typename F>
inline void binary_into_buffer_optimised(
        S* RPY_RESTRICT dptr, const S* RPY_RESTRICT lptr,
        const S* RPY_RESTRICT rptr, const dimn_t count,
        const dimn_t block_count, const uint64_t* mask, F&& func
)
{
    for (dimn_t block = 0; block < block_count; ++block) {
        auto block_mask = (mask == nullptr) ? ~dimn_t(0) : mask[block];
        // 64*block is always a valid address while block < block_count
        const auto start_idx = 64 * block;
        const auto number = std::min(count - start_idx, dimn_t(64));

        auto* bdptr = dptr + start_idx;

        /*
         * Hopefully the compiler will do 2 things here:
         *  - optimise away the call to func;
         *  - replace this inner loop with vector add + masked store
         *  instructions. Since the block is 64 elements wide it should be
         *  able to do a pretty good job with this.
         */
        if (lptr == nullptr) {
            const auto* brptr = rptr + start_idx;

            for (dimn_t i = 0; i < number; ++i) {
                if (block_mask & 1) { bdptr[i] = func(bdptr[i], brptr[i]); }
                block_mask >>= 1;
            }
        } else if (rptr == nullptr) {
            const auto* blptr = lptr + start_idx;
            for (dimn_t i = 0; i < number; ++i) {
                if (block_mask & 1) { bdptr[i] = func(blptr[i], bdptr[i]); }
                block_mask >>= 1;
            }
        } else {
            const auto* blptr = lptr + start_idx;
            const auto* brptr = rptr + start_idx;

            for (dimn_t i = 0; i < number; ++i) {
                if (block_mask & 1) { bdptr[i] = func(blptr[i], brptr[i]); }
                block_mask >>= 1;
            }
        }
    }
}

template <typename S, typename F>
inline void binary_into_buffer(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask, F&& func
)
{
    if (count == 0) { return; }

    RPY_CHECK(!dst.is_null());

    // dst type is the reference type, otherwise the caller should have done
    // something different.
    const auto* type = dst.type();
    RPY_DBG_ASSERT(type == ScalarType::of<S>());

    // Round up
    auto block_count = (count + 63) / 64;

    auto* dptr = dst.template raw_cast<S*>();

    if (lhs.type() == type && rhs.type() == type) {
        // Use the optimised version above.

        // Raw cast is just a reinterpret cast for const types
        const auto* lptr = lhs.template raw_cast<const S*>();
        const auto* rptr = rhs.template raw_cast<const S*>();
        binary_into_buffer_optimised(
                dptr, lptr, rptr, count, block_count, mask,
                std::forward<F>(func)
        );
        return;
    }

    /*
     * What we want to do here is essentially the following loop
     *
     * for (dimn_t i=0; i<count; ++i) {
     *     if (mask[i/64] & 1) {
     *         dptr[i] = func(try_convert(lptr[i], rptr[i]));
     *     }
     * }
     *
     * but this is going to be horrendously inefficient. Since we anyway need to
     * block by the mask size (64), we can use this to our advantage and
     * temporarily convert_copy 64 values at a time, and then apply the function
     * as if everything were native.
     */

    for (dimn_t block_idx = 0; block_idx < block_count; ++block_idx) {
        const auto block_start = 64 * block_idx;
        const auto block_size = std::min(count - 64 * block_idx, dimn_t(64));
        auto block_mask = (mask == nullptr) ? ~dimn_t(0) : mask[block_idx];

        auto* block_dst = dptr + block_start;

        std::vector<S> lbuffer(64);
        std::vector<S> rbuffer(64);
        if (!lhs.is_null()) {
            type->convert_copy(
                    {type, lbuffer.data()}, lhs + block_start, block_size
            );
        } else {
            lbuffer.assign(block_dst, block_dst + block_size);
        }
        if (!rhs.is_null()) {
            type->convert_copy(
                    {type, rbuffer.data()}, rhs + block_start, block_size
            );
        } else {
            rbuffer.assign(block_dst, block_dst + block_size);
        }

        /*
         * The beauty of this approach is that it doesn't matter if either of
         * lhs_ptr or rhs_ptr are null, since we always fill it with the
         * correct values for that case. This is a bit wasteful, but it will
         * always work.
         */
        for (dimn_t i = 0; i < block_size; ++i) {
            if (block_mask & 1) { block_dst[i] = func(lbuffer[i], rbuffer[i]); }
            block_mask >>= 1;
        }
    }
}

}// namespace impl_helpers
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_TYPE_HELPER_H_
