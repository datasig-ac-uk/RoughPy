//
// Created by sam on 25/06/24.
//

#include "scalar_array.h"

using namespace rpy;
using namespace rpy::scalars;

const devices::Buffer& ScalarArray::buffer() const { return *this; }
devices::Buffer& ScalarArray::mut_buffer()
{
    RPY_CHECK(!is_const());
    return *this;
}
void ScalarArray::check_for_ptr_access(bool mut) const
{
    RPY_CHECK(is_host());
    RPY_CHECK(!mut || mode() != devices::BufferMode::Read);
}
ScalarArray ScalarArray::borrow() const { return *this; }

ScalarArray ScalarArray::borrow_mut()
{
    RPY_CHECK(!is_const());
    return *this;
}

ScalarCRef ScalarArray::operator[](dimn_t i) const
{
    RPY_CHECK(i < size() && is_host());
    const TypePtr tp = type();
    const auto* p = static_cast<const byte*>(ptr()) + i * size_of(*tp);
    return ScalarCRef(std::move(tp), p);
}

ScalarRef ScalarArray::operator[](dimn_t i)
{
    RPY_CHECK(i < size() && is_host() && !is_const());
    const TypePtr tp = type();
    auto* p = static_cast<byte*>(ptr()) + i * size_of(*tp);
    return ScalarRef(std::move(tp), p);
}

ScalarArray ScalarArray::operator[](SliceIndex index)
{
    RPY_DBG_ASSERT(index.begin < index.end);
    const auto buffer_size = size();
    RPY_CHECK(
            index.end <= buffer_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(buffer_size)
    );

    const auto offset = index.begin;
    const auto sz = (index.end - index.begin);

    return ScalarArray(slice(offset, sz));
}

ScalarArray ScalarArray::operator[](SliceIndex index) const
{
    RPY_DBG_ASSERT(index.begin <= index.end);
    const auto buffer_size = size();
    RPY_CHECK(
            index.end <= buffer_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(buffer_size)
    );
    const auto offset = index.begin;
    const auto sz = (index.end - index.begin);

    return ScalarArray(slice(offset, sz));
}

ScalarArray ScalarArray::to_device(devices::Device device) const
{
    if (device == this->device()) { return *this; }
    auto new_buffer = device->alloc(*this->type(), this->size());
    Buffer::to_device(new_buffer, device);
    return ScalarArray(std::move(new_buffer));
}
