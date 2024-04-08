//
// Created by sam on 3/31/24.
//

#ifndef ROUGHPY_DEVICES_ALGORITHMS_H
#define ROUGHPY_DEVICES_ALGORITHMS_H

#include <roughpy/scalars/scalars_fwd.h>

#include "buffer.h"
#include "core.h"
#include "device_handle.h"
#include "type.h"
#include "value.h"

#include <roughpy/core/alloc.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <boost/smart_ptr/intrusive_ref_counter.hpp>

namespace rpy {
namespace devices {

/**
 * @class AlgorithmDrivers
 * @brief The AlgorithmDrivers class provides a set of algorithms that can be
 * applied to buffers.
 *
 * The AlgorithmDrivers class is an abstract base class that provides a common
 * interface for implementing algorithms that operate on buffers.
 *
 * @sa Buffer
 */
class ROUGHPY_SCALARS_EXPORT AlgorithmDrivers
    : public boost::intrusive_ref_counter<AlgorithmDrivers>
{
public:
    virtual ~AlgorithmDrivers();

    /**
     * @brief Find the first position where a value occurs in a buffer.
     *
     * This method is used to find the position of the first occurrence
     * of a specified value in a buffer. It returns an optional that contains
     * the position if the value is found, or an optional without a value if the
     * value is not found.
     *
     * @param buffer The buffer in which to search for the value.
     * @param value The value to search for.
     *
     * @return An optional containing the position of the first occurrence of
     * the specified value in the buffer. If the value is not found, an optional
     * without a value is returned.
     *
     * @sa Buffer
     */
    RPY_NO_DISCARD virtual optional<dimn_t>
    find(const Buffer& buffer, Reference value) const;

    RPY_NO_DISCARD virtual dimn_t
    count(const Buffer& buffer, Reference value) const;

    /**
     * @brief Find the first position where two buffers differ.
     *
     * This method compares the elements of the left and right buffers and
     * returns the position of the first mismatch. If the buffers are equal, it
     * returns an optional that does not contain a value.
     *
     * @param left The first buffer to compare.
     * @param right The second buffer to compare.
     *
     * @return An optional containing the position of the first mismatch if the
     * buffers are different. If the buffers are equal, an optional without a
     * value is returned.
     *
     * @sa Buffer
     */
    RPY_NO_DISCARD virtual optional<dimn_t>
    mismatch(const Buffer& left, const Buffer& right) const;

    /**
     * @brief Copy the contents of one buffer to another.
     *
     * This method copies the contents of the source buffer to the destination
     * buffer. The size and capacity of the destination buffer will be adjusted
     * to match the source buffer.
     *
     * @param dest The destination buffer to copy to.
     * @param source The source buffer from which to copy.
     *
     * @note The source buffer is not modified.
     *
     * @warning The destination buffer must have enough capacity to hold the
     * entire contents of the source buffer. Otherwise, undefined behavior may
     * occur.
     *
     * @sa Buffer
     */
    virtual void copy(Buffer& dest, const Buffer& source) const;

    virtual void swap_ranges(Buffer& left, Buffer& right) const;

    virtual void fill(Buffer& dst, Reference value) const;

    virtual void reverse(Buffer& buffer) const;

    virtual void shift_left(Buffer& buffer) const;

    virtual void shift_right(Buffer& buffer) const;

    virtual optional<dimn_t>
    lower_bound(const Buffer& buffer, Reference value) const;

    virtual optional<dimn_t>
    upper_bound(const Buffer& buffer, Reference value) const;

    virtual void max(const Buffer& buffer, Reference out) const;

    virtual void min(const Buffer& buffer, Reference out) const;

    virtual bool
    lexicographacal_compare(const Buffer& left, const Buffer& rought) const;
};


namespace algorithms {

RPY_NO_DISCARD inline optional<dimn_t>
find(const Buffer& buffer, Reference value)
{
    if (buffer.is_null()) { return {}; }
    auto algo = buffer.device()->algorithms(buffer.content_type());
    return algo->find(buffer, value);
}

RPY_NO_DISCARD inline optional<dimn_t>
lower_bound(const Buffer& buffer, Reference value)
{
    if (buffer.is_null()) { return {}; }
    auto algo = buffer.device()->algorithms(buffer.content_type());
    return algo->lower_bound(buffer, value);
}

RPY_NO_DISCARD inline optional<dimn_t>
upper_bound(const Buffer& buffer, Reference value)
{
    if (buffer.is_null()) { return {}; }
    auto algo = buffer.device()->algorithms(buffer.content_type());
    return algo->lower_bound(buffer, value);
}

RPY_NO_DISCARD inline dimn_t count(const Buffer& buffer, Reference value)
{
    if (buffer.is_null()) { return 0; }

    auto algo = buffer.device()->algorithms(buffer.content_type());
    return algo->count(buffer, value);
}

RPY_NO_DISCARD inline bool contains(const Buffer& buffer, Reference value)
{
    return static_cast<bool>(find(buffer, value));
}

RPY_NO_DISCARD inline optional<dimn_t>
mismatch(const Buffer& left, const Buffer& right)
{
    if (left.is_null()) {
        if (right.is_null()) { return {}; }
        return 0;
    }
    if (right.is_null()) { return 0; }

    Device devices[] = {left.device(), right.device()};
    const auto device = get_best_device(devices);

    auto algo = device->algorithms(left.content_type(), right.content_type());
    return algo->mismatch(left, right);
}

RPY_NO_DISCARD inline bool equal(const Buffer& left, const Buffer& right)
{
    return static_cast<bool>(mismatch(left, right));
}

RPY_NO_DISCARD inline bool lexicographical_compare(const Buffer& left, const Buffer& right)
{
    if (left.is_null()) {
        if (right.is_null()) { return {}; }
        return false;
    }
    if (right.is_null()) { return false; }

    Device devices[] = {left.device(), right.device()};
    const auto device = get_best_device(devices);

    auto algo = device->algorithms(left.content_type(), right.content_type());
    return algo->lexicographacal_compare(left, right);
}



inline void copy(Buffer& dst, const Buffer& src)
{
    if (src.is_null()) {
        return;
    }

    Device device;
    const auto* src_type = src.content_type();
    if (dst.is_null()) {
        device = src.device();
        dst = src_type->allocate(device, src.size());
    } else {
        Device devices[] = { dst.device(), src.device() };
        RPY_CHECK(src.size() <= dst.size());
        device = get_best_device(devices);
    }

    auto algo = device->algorithms(dst.content_type(), src_type, true);;

    return algo->copy(dst, src);
}


inline void swap_ranges(Buffer& left, Buffer& right)
{
    if (left.is_null() && right.is_null()) {
        return;
    }
    RPY_CHECK(!left.is_null() && !right.is_null());

    RPY_CHECK(left.size() == right.size());
    const auto left_type = left.content_type();
    RPY_CHECK(left_type == right.content_type());

    Device devices[] = { left.device(), right.device()};
    Device device = get_best_device(devices);

    auto algo = device->algorithms(left_type);
    algo->swap_ranges(left, right);
}

inline void reverse(Buffer& buffer)
{
    if (buffer.empty()) { return; }

    auto algo = buffer.device()->algorithms(buffer.content_type());
    algo->reverse(buffer);
}

inline void shift_left(Buffer& buffer)
{
     if (buffer.empty()) { return; }

    auto algo = buffer.device()->algorithms(buffer.content_type());
    algo->shift_left(buffer);
}


inline void shift_right(Buffer& buffer)
{
     if (buffer.empty()) { return; }

    auto algo = buffer.device()->algorithms(buffer.content_type());
    algo->shift_left(buffer);
}

inline void fill(Buffer& buffer, Reference value)
{
    if (buffer.empty()) {
        return;
    }

    auto algo = buffer.device()->algorithms(buffer.content_type());
    algo->fill(buffer, value);
}

inline void min(const Buffer& buffer, Reference out)
{
    if (buffer.empty()) { return; }

    auto algo = buffer.device()->algorithms(buffer.content_type());
    algo->min(buffer, out);
}

inline void max(const Buffer& buffer, Reference out)
{
    if (buffer.empty()) { return; }

    auto algo = buffer.device()->algorithms(buffer.content_type());
    algo->max(buffer, out);
}

}// namespace algorithms

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_ALGORITHMS_H
