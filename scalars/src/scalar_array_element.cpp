//
// Created by sam on 3/12/24.
//

#include "scalar_array_element.h"
#include "scalar.h"
#include "scalar/arithmetic.h"
#include "scalar/print.h"

using namespace rpy;
using namespace rpy::scalars;

ScalarArrayElement::ScalarArrayElement(
        devices::Buffer& buffer,
        dimn_t index,
        const ScalarType* type
)
    : m_buffer(buffer.map(buffer.size(), 0)),
      m_index(index)
{
    if (type != nullptr) {
        p_type_or_info = dtl::pack_type(type);
    } else {
        p_type_or_info = dtl::pack_type(buffer.type_info());
    }
}
ScalarArrayElement::ScalarArrayElement(
        const devices::Buffer& buffer,
        dimn_t index,
        const ScalarType* type
)
    : m_buffer(buffer.map(buffer.size(), 0)),
      m_index(index)
{

    if (type != nullptr) {
        p_type_or_info = dtl::pack_type(type);
    } else {
        p_type_or_info = dtl::pack_type(buffer.type_info());
    }
}

void* ScalarArrayElement::mut_pointer()
{
    RPY_CHECK(m_buffer.mode() == devices::BufferMode::Write);
    const auto info = type_info_from(p_type_or_info);
    return static_cast<byte*>(m_buffer.ptr()) + m_index * info.bytes;
}
const void* ScalarArrayElement::pointer() const noexcept
{
    const auto info = type_info_from(p_type_or_info);
    return static_cast<const byte*>(m_buffer.ptr()) + m_index * info.bytes;
}
void ScalarArrayElement::set_value(const Scalar& value)
{
    RPY_CHECK(m_buffer.mode() == devices::BufferMode::Write);
    const auto info = type_info_from(p_type_or_info);
    auto* ptr = static_cast<byte*>(m_buffer.ptr()) + m_index * info.bytes;
    scalars::dtl::scalar_convert_copy(ptr, info, value);
}
void ScalarArrayElement::print(std::ostream& os) const
{
    dtl::print_scalar_val(os, pointer(), type_info_from(p_type_or_info));
}
void ScalarArrayElement::add_inplace(const Scalar& other)
{
    dtl::scalar_inplace_add(
            mut_pointer(),
            p_type_or_info,
            other.pointer(),
            other.packed_type_info()
    );
}
void ScalarArrayElement::sub_inplace(const Scalar& other)
{
    dtl::scalar_inplace_sub(
            mut_pointer(),
            p_type_or_info,
            other.pointer(),
            other.packed_type_info()
    );
}
void ScalarArrayElement::mul_inplace(const Scalar& other)
{
    dtl::scalar_inplace_mul(
            mut_pointer(),
            p_type_or_info,
            other.pointer(),
            other.packed_type_info()
    );
}
void ScalarArrayElement::div_inplace(const Scalar& other)
{
    dtl::scalar_inplace_div(
            mut_pointer(),
            p_type_or_info,
            other.pointer(),
            other.packed_type_info()
    );
}
