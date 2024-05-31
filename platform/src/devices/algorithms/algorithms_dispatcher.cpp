//
// Created by sam on 5/16/24.
//

#include "algorithms.h"

using namespace rpy;
using namespace rpy::devices;

optional<dimn_t>
AlgorithmsDispatcher::find(const Buffer& buffer, ConstReference value) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    return impl.find(buffer, value);
}

dimn_t
AlgorithmsDispatcher::count(const Buffer& buffer, ConstReference value) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    return impl.count(buffer, value);
}

optional<dimn_t>
AlgorithmsDispatcher::mismatch(const Buffer& left, const Buffer& right) const
{
    auto& impl
            = this->get_implementor(left.type(), right.type());
    return impl.mismatch(left, right);
}

void AlgorithmsDispatcher::copy(Buffer& dest, const Buffer& source) const
{
    auto& impl
            = this->get_implementor(dest.type(), source.type());
    impl.copy(dest, source);
}

void AlgorithmsDispatcher::swap_ranges(Buffer& left, Buffer& right) const
{
    auto& impl
            = this->get_implementor(left.type(), right.type());
    impl.swap_ranges(left, right);
}

void AlgorithmsDispatcher::fill(Buffer& dst, ConstReference value) const
{
    auto& impl = this->get_implementor(dst.type(), dst.type());
    impl.fill(dst, value);
}

void AlgorithmsDispatcher::reverse(Buffer& buffer) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    impl.reverse(buffer);
}
void AlgorithmsDispatcher::shift_left(Buffer& buffer) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    impl.shift_left(buffer);
}

void AlgorithmsDispatcher::shift_right(Buffer& buffer) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    impl.shift_right(buffer);
}

optional<dimn_t> AlgorithmsDispatcher::lower_bound(
        const Buffer& buffer,
        ConstReference value
) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    return impl.lower_bound(buffer, value);
}

optional<dimn_t> AlgorithmsDispatcher::upper_bound(
        const Buffer& buffer,
        ConstReference value
) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    return impl.upper_bound(buffer, value);
}

void AlgorithmsDispatcher::max(const Buffer& buffer, Reference out) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    impl.max(buffer, out);
}

void AlgorithmsDispatcher::min(const Buffer& buffer, Reference out) const
{
    auto& impl = this->get_implementor(
            buffer.type(),
            buffer.type()
    );
    impl.min(buffer, out);
}

bool AlgorithmsDispatcher::lexicographical_compare(
        const Buffer& left,
        const Buffer& right
) const
{
    auto& impl
            = this->get_implementor(left.type(), right.type());
    return impl.lexicographical_compare(left, right);
}
