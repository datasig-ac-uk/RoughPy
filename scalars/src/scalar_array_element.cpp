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
      m_index(index),
      m_info(buffer.type_info()),
      p_type(type)
{
    if (buffer.is_host()) {
        m_buffer = buffer;
    } else {
        m_buffer = buffer.map(buffer.size(), 0);
    }
}
ScalarArrayElement::ScalarArrayElement(
        const devices::Buffer& buffer,
        dimn_t index,
        const ScalarType* type
)
    : m_buffer(buffer.map(buffer.size(), 0)),
      m_index(index),
      m_info(buffer.type_info()),
      p_type(type)
{}

void* ScalarArrayElement::mut_pointer()
{
    RPY_CHECK(m_buffer.mode() == devices::BufferMode::Write);
    return static_cast<byte*>(m_buffer.ptr()) + m_index * m_info.bytes;
}
const void* ScalarArrayElement::pointer() const noexcept
{
    return static_cast<const byte*>(m_buffer.ptr()) + m_index * m_info.bytes;
}
void ScalarArrayElement::set_value(const Scalar& value)
{
    RPY_CHECK(m_buffer.mode() == devices::BufferMode::Write);
    auto* ptr = static_cast<byte*>(m_buffer.ptr()) + m_index * m_info.bytes;
    scalars::dtl::scalar_convert_copy(ptr, m_info, value);
}
void ScalarArrayElement::print(std::ostream& os) const
{
    dtl::print_scalar_val(os, pointer(), m_info);
}
void ScalarArrayElement::add_inplace(const Scalar& other)
{
    auto* dst_ptr = mut_pointer();

    if (p_type != nullptr) {
        dtl::scalar_inplace_add(
                dst_ptr,
                p_type,
                other.pointer(),
                other.packed_type_info()
        );
    } else {
        dtl::scalar_inplace_add(
                dst_ptr,
                m_info,
                other.pointer(),
                other.packed_type_info()
        );
    }
}
void ScalarArrayElement::sub_inplace(const Scalar& other)
{
    auto* dst_ptr = mut_pointer();

    if (p_type != nullptr) {
        dtl::scalar_inplace_sub(
                dst_ptr,
                p_type,
                other.pointer(),
                other.packed_type_info()
        );
    } else {
        dtl::scalar_inplace_sub(
                dst_ptr,
                m_info,
                other.pointer(),
                other.packed_type_info()
        );
    }
}
void ScalarArrayElement::mul_inplace(const Scalar& other)
{
    auto* dst_ptr = mut_pointer();

    if (p_type != nullptr) {
        dtl::scalar_inplace_mul(
                dst_ptr,
                p_type,
                other.pointer(),
                other.packed_type_info()
        );
    } else {
        dtl::scalar_inplace_mul(
                dst_ptr,
                m_info,
                other.pointer(),
                other.packed_type_info()
        );
    }
}
void ScalarArrayElement::div_inplace(const Scalar& other)
{
    auto* dst_ptr = mut_pointer();

    if (p_type != nullptr) {
        dtl::scalar_inplace_div(
                dst_ptr,
                p_type,
                other.pointer(),
                other.packed_type_info()
        );
    } else {
        dtl::scalar_inplace_div(
                dst_ptr,
                m_info,
                other.pointer(),
                other.packed_type_info()
        );
    }
}
