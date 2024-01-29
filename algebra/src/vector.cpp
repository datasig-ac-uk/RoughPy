//
// Created by sam on 1/29/24.
//

#include "vector.h"

#include <sstream>

using namespace rpy;
using namespace algebra;

namespace rpy {
namespace algebra {

class VectorIterator
{
};

class BasisKey
{
};

}// namespace algebra
}// namespace rpy

void Vector::resize_dim(rpy::deg_t dim)
{

    if (is_sparse()) {}
}

void Vector::resize_degree(rpy::deg_t degree) {}

dimn_t Vector::dimension() const noexcept { return m_scalar_buffer.size(); }

dimn_t Vector::size() const noexcept {}

bool Vector::is_zero() const noexcept {}

void Vector::make_dense() {}
void Vector::make_sparse() {}
scalars::Scalar Vector::get(BasisKey key) const { return scalars::Scalar(); }
scalars::Scalar Vector::get_mut(BasisKey key) { return scalars::Scalar(); }
Vector::iterator Vector::begin() noexcept
{
    return rpy::algebra::Vector::iterator();
}
Vector::iterator Vector::end() noexcept
{
    return rpy::algebra::Vector::iterator();
}
Vector::const_iterator Vector::begin() const noexcept
{
    return rpy::algebra::Vector::const_iterator();
}
Vector::const_iterator Vector::end() const noexcept
{
    return rpy::algebra::Vector::const_iterator();
}
scalars::Scalar Vector::operator[](BasisKey key) const
{
    return scalars::Scalar();
}
scalars::Scalar Vector::operator[](BasisKey key) { return scalars::Scalar(); }

devices::Kernel
Vector::get_kernel(OperationType type, string_view operation) const
{
    RPY_DBG_ASSERT(!operation.empty());
    const auto* stype = scalar_type();

    string op(operation);
    op += '_';
    op += stype->id();

    if (type != Unary && type != UnaryInplace) {
        op += (is_sparse() ? "_sparse" : "_dense");
    }

    auto kernel = stype->device()->get_kernel(op);
    RPY_CHECK(kernel);
    return *kernel;
}

Vector Vector::uminus() const { return Vector(); }
Vector Vector::add(const Vector& other) const { return Vector(); }
Vector Vector::sub(const Vector& other) const { return Vector(); }
Vector Vector::left_smul(const scalars::Scalar& other) const
{
    return Vector();
}
Vector Vector::right_smul(const scalars::Scalar& other) const
{
    return Vector();
}
Vector Vector::sdiv(const scalars::Scalar& other) const { return Vector(); }
Vector& Vector::add_inplace(const Vector& other) { return *this; }
Vector& Vector::sub_inplace(const Vector& other) { return *this; }
Vector& Vector::smul_inplace(const Vector& other) { return *this; }
Vector& Vector::sdiv_inplace(const Vector& other) { return *this; }
Vector& Vector::add_scal_mul(const Vector& other, const scalars::Scalar& scalar)
{
    return *this;
}
Vector& Vector::sub_scal_mul(const Vector& other, const scalars::Scalar& scalar)
{
    return *this;
}
Vector& Vector::add_scal_div(const Vector& other, const scalars::Scalar& scalar)
{
    return *this;
}
Vector& Vector::sub_scal_div(const Vector& other, const scalars::Scalar& scalar)
{
    return *this;
}
bool Vector::operator==(const Vector& other) const { return false; }

std::ostream& algebra::operator<<(std::ostream& os, const Vector& value)
{
    return <#initializer #>;
}
