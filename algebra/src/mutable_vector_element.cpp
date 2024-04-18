//
// Created by sam on 21/03/24.
//

#include "mutable_vector_element.h"
#include "vector.h"

using namespace rpy;
using namespace rpy::algebra;

void MutableVectorElement::get_value()
{
    if (auto index = p_vector->get_index(m_key)) {
        m_loc = index;
    } else {
        m_value = scalars::Scalar(p_vector->scalar_type());
    }
}
MutableVectorElement::~MutableVectorElement()
{
    if (m_value.is_zero()) {
        if (m_loc) { p_vector->delete_element(m_key, m_loc); }
    } else {
        if (!m_loc) {}
    }
}

scalars::PackedScalarType MutableVectorElement::type() const noexcept
{
    return p_vector->scalar_type();
}

const void* MutableVectorElement::pointer() const noexcept { return nullptr; }
void MutableVectorElement::set_value(const scalars::Scalar& value) {}
void MutableVectorElement::print(std::ostream& os) const {}
void MutableVectorElement::add_inplace(const scalars::Scalar& other)
{
    ScalarInterface::add_inplace(other);
}
void MutableVectorElement::sub_inplace(const scalars::Scalar& other)
{
    ScalarInterface::sub_inplace(other);
}
void MutableVectorElement::mul_inplace(const scalars::Scalar& other)
{
    ScalarInterface::mul_inplace(other);
}
void MutableVectorElement::div_inplace(const scalars::Scalar& other)
{
    ScalarInterface::div_inplace(other);
}
