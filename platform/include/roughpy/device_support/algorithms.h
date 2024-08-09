//
// Created by sam on 5/16/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_ALGORITHMS_H
#define ROUGHPY_DEVICE_SUPPORT_ALGORITHMS_H

#include <roughpy/core/container/unordered_map.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/smart_ptr.h>
#include <roughpy/core/strings.h>

#include <roughpy/devices/algorithms.h>
#include <roughpy/devices/type.h>
#include <roughpy/devices/type_dispatcher.h>

#include "algorithm_drivers/copy.h"
#include "algorithm_drivers/count.h"
#include "algorithm_drivers/default_fill.h"
#include "algorithm_drivers/fill.h"
#include "algorithm_drivers/find.h"
#include "algorithm_drivers/lexicographical_compare.h"
#include "algorithm_drivers/lower_bound.h"
#include "algorithm_drivers/max_element.h"
#include "algorithm_drivers/min_element.h"
#include "algorithm_drivers/mismatch.h"
#include "algorithm_drivers/reverse.h"
#include "algorithm_drivers/rotate.h"
#include "algorithm_drivers/shift_left.h"
#include "algorithm_drivers/shift_right.h"
#include "algorithm_drivers/swap_ranges.h"
#include "algorithm_drivers/upper_bound.h"

namespace rpy {
namespace devices {

template <typename S, typename T = S>
class HostDriversImpl : public AlgorithmDrivers
{
    TypePtr p_primary_type = get_type<S>();
    TypePtr p_secondary_type = get_type<T>();

public:
    explicit HostDriversImpl(std::tuple<TypePtr, TypePtr> types)
        : p_primary_type(std::move(std::get<0>(types))),
          p_secondary_type(std::move(std::get<1>(types)))
    {}

    pair<string_view, string_view> get_index() const noexcept
    {
        return {p_primary_type->id(), p_secondary_type->id()};
    }

    void default_fill(Buffer& buffer) const override;
    RPY_NO_DISCARD optional<dimn_t>
    find(const Buffer& buffer, ConstReference value) const override;
    RPY_NO_DISCARD dimn_t
    count(const Buffer& buffer, ConstReference value) const override;
    RPY_NO_DISCARD optional<dimn_t>
    mismatch(const Buffer& left, const Buffer& right) const override;
    void copy(Buffer& dest, const Buffer& source) const override;
    void swap_ranges(Buffer& left, Buffer& right) const override;
    void fill(Buffer& dst, ConstReference value) const override;
    void reverse(Buffer& buffer) const override;
    void shift_left(Buffer& buffer) const override;
    void shift_right(Buffer& buffer) const override;
    optional<dimn_t>
    lower_bound(const Buffer& buffer, ConstReference value) const override;
    optional<dimn_t>
    upper_bound(const Buffer& buffer, ConstReference value) const override;
    void max(const Buffer& buffer, Reference out) const override;
    void min(const Buffer& buffer, Reference out) const override;
    bool lexicographical_compare(const Buffer& left, const Buffer& right)
            const override;
};

template <typename S, typename T>
void HostDriversImpl<S, T>::default_fill(Buffer& buffer) const
{
    constexpr auto func = default_fill_func<S>;
    func.type_check(*p_primary_type, *buffer.type());
    return func(buffer);
}
template <typename S, typename T>
optional<dimn_t>
HostDriversImpl<S, T>::find(const Buffer& buffer, ConstReference value) const
{
    constexpr auto func = find_func<S, T>;
    func.type_check(*p_primary_type, *buffer.type());
    func.type_check(*p_secondary_type, *value.type());
    return func(buffer, value);
}
template <typename S, typename T>
dimn_t
HostDriversImpl<S, T>::count(const Buffer& buffer, ConstReference value) const
{
    constexpr auto func = count_func<S, T>;
    func.type_check(*p_primary_type, *buffer.type());
    func.type_check(*p_secondary_type, *value.type());
    return func(buffer, value);
}

template <typename S, typename T>
void HostDriversImpl<S, T>::copy(
        Buffer& destination_buffer,
        const Buffer& source_buffer
) const
{
    constexpr auto func = copy_func<S, T>;
    func.type_check(*p_primary_type, *destination_buffer.type());
    func.type_check(*p_secondary_type, *source_buffer.type());
    func(destination_buffer, source_buffer);
}

template <typename S, typename T>
void HostDriversImpl<S, T>::swap_ranges(
        Buffer& left_buffer,
        Buffer& right_buffer
) const
{
    constexpr auto func = swap_ranges_func<S, T>;
    func.type_check(*p_primary_type, *left_buffer.type());
    func.type_check(*p_primary_type, *left_buffer.type());
    func(left_buffer, right_buffer);
}

template <typename S, typename T>
void HostDriversImpl<S, T>::fill(
        Buffer& destination_buffer,
        ConstReference value
) const
{
    constexpr auto func = fill_func<S, T>;
    func.type_check(*p_primary_type, *destination_buffer.type());
    func.type_check(*p_secondary_type, *value.type());
    return func(destination_buffer, value);
}

template <typename S, typename T>
void HostDriversImpl<S, T>::reverse(Buffer& buffer) const
{
    constexpr auto func = reverse_func<S, T>;
    func.type_check(*p_primary_type, *buffer.type());
    func(buffer);
}

template <typename S, typename T>
void HostDriversImpl<S, T>::shift_left(Buffer& buffer) const
{
    constexpr auto func = rotate_func<S, T>;
    func.type_check(*p_primary_type, *buffer.type());
    func(buffer, 1);
}

template <typename S, typename T>
void HostDriversImpl<S, T>::shift_right(Buffer& buffer) const
{
    // RPY_CHECK(
    //         p_primary_type->compare_with(buffer.content_type())
    //         == TypeComparison::AreSame
    // );
    // auto buffer_view = buffer.map();
    // auto buffer_slice = buffer_view.as_mut_slice<S>();
    //
    // ranges::rotate(buffer_slice | views::reverse, buffer_slice.rbegin() + 1);
}

template <typename S, typename T>
optional<dimn_t>
HostDriversImpl<S, T>::lower_bound(const Buffer& buffer, ConstReference value)
        const
{
    constexpr auto func = lower_bound_func<S, T>;
    func.type_check(*p_primary_type, *buffer.type());
    func.type_check(*p_secondary_type, *value.type());
    return func(buffer, value);
}

template <typename S, typename T>
optional<dimn_t>
HostDriversImpl<S, T>::upper_bound(const Buffer& buffer, ConstReference value)
        const
{
    constexpr auto func = upper_bound_func<S, T>;
    func.type_check(*p_primary_type, *buffer.type());
    func.type_check(*p_secondary_type, *value.type());
    return func(buffer, value);
}

template <typename S, typename T>
optional<dimn_t>
HostDriversImpl<S, T>::mismatch(const Buffer& left, const Buffer& right) const
{
    constexpr auto func = mismatch_func<S, T>;
    func.type_check(*p_primary_type, *left.type());
    func.type_check(*p_secondary_type, *right.type());
    return func(left, right);
}

template <typename S, typename T>
void HostDriversImpl<S, T>::max(const Buffer& buffer, Reference out) const
{
    constexpr auto func = max_element_func<S, T>;
    func.type_check(*p_primary_type, *buffer.type());
    func.type_check(*p_secondary_type, *out.type());
    func(buffer, out);
}

template <typename S, typename T>
void HostDriversImpl<S, T>::min(const Buffer& buffer, Reference out) const
{
    constexpr auto func = min_element_func<S, T>;
    func.type_check(*p_primary_type, *buffer.type());
    func.type_check(*p_secondary_type, *out.type());
    func(buffer, out);
}

template <typename S, typename T>
bool HostDriversImpl<S, T>::lexicographical_compare(
        const Buffer& left_buffer,
        const Buffer& right_buffer
) const
{
    constexpr auto func = lexicographical_compare_func<S, T>;
    func.type_check(*p_primary_type, *left_buffer.type());
    func.type_check(*p_secondary_type, *right_buffer.type());
    return func(left_buffer, right_buffer);
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_ALGORITHMS_H
