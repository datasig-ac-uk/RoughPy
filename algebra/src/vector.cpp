//
// Created by sam on 1/29/24.
//

#include "vector.h"

#include "basis_key.h"


#include <algorithm>
#include <sstream>

using namespace rpy;
using namespace algebra;


namespace rpy {
namespace algebra {

class VectorIterator {};

}
}

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


devices::KernelLaunchParams Vector::get_kernel_launch_params() const
{
    return devices::KernelLaunchParams();
}


void Vector::resize_dim(rpy::deg_t dim)
{

    if (is_sparse()) {}
}

void Vector::resize_degree(rpy::deg_t degree) {}

dimn_t Vector::dimension() const noexcept { return m_scalar_buffer.size(); }

dimn_t Vector::size() const noexcept {}

bool Vector::is_zero() const noexcept
{
    if (fast_is_zero()) {
        return true;
    }

    auto kernel = get_kernel(Unary, "all_equal");
    auto params = get_kernel_launch_params();



    return true;
}

void Vector::set_zero()
{
    if (is_sparse()) {
        m_scalar_buffer = scalars::ScalarArray(scalar_type());
        m_key_buffer = devices::Buffer();
    } else {
        auto kernel = get_kernel(UnaryInplace, "set_scalar");
        auto params = get_kernel_launch_params();

        kernel(params, m_scalar_buffer.mut_buffer(), 0);
    }
}


optional <dimn_t> Vector::get_index(rpy::algebra::BasisKey key) const noexcept
{

    if (is_dense() && key.is_index()) {
        auto index = key.get_index();
        return index >= dimension() ? optional<dimn_t>() : index;
    }

    if (!p_basis->has_key(key)) {
        return {};
    }

    const auto* basis = p_basis.get();
    if (p_basis->is_ordered()) {
        if (is_dense()) {
            auto index = p_basis->to_index(key);
            return index >= dimension() ? optional<dimn_t>() : index;
        }

        auto keys = m_key_buffer.as_slice<BasisKey>();

        auto compare = [basis](const BasisKey& lhs, const BasisKey& rhs) {
            return basis->less(lhs, rhs);
        };

        const auto* begin = keys.begin();
        const auto* end = keys.end();
        const auto* found = std::lower_bound(begin, end, key, compare);
        if (found != end && p_basis->equals(key, *found)) {
            return static_cast<dimn_t>(found - begin);
        }
    } else {
        auto keys = m_key_buffer.as_slice<BasisKey>();
        const auto* begin = keys.begin();
        const auto* end = keys.end();

        auto compare = [basis, key](const BasisKey& arg) {
            return basis->equals(arg, key);
        };
        const auto* found = std::find_if(begin, end, compare);

        if (found != end) {
            return static_cast<dimn_t>(found - begin);
        }
    }

    return {};
}


void Vector::make_dense() {}

void Vector::make_sparse() {}


Vector::Vector() : p_basis(), m_scalar_buffer(), m_key_buffer()
{
}

Vector::~Vector() = default;

Vector::Vector(const Vector& other)
    : p_basis(other.p_basis), m_scalar_buffer(other.m_scalar_buffer),
      m_key_buffer(other.m_key_buffer.clone())
{
}

Vector::Vector(Vector&& other) noexcept
    : p_basis(std::move(other.p_basis)),
      m_scalar_buffer(std::move(other.m_scalar_buffer)),
      m_key_buffer(std::move(other.m_key_buffer))
{
}

Vector& Vector::operator=(const Vector& other)
{
    if (&other != this) {
        this->~Vector();
        p_basis = other.p_basis;
        m_scalar_buffer = other.m_scalar_buffer;
        m_key_buffer = other.m_key_buffer.clone();
    }
    return *this;
}

Vector& Vector::operator=(Vector&& other) noexcept
{
    if (&other != this) {
        this->~Vector();
        p_basis = std::move(other.p_basis);
        m_scalar_buffer = std::move(other.m_scalar_buffer);
        m_key_buffer = std::move(other.m_key_buffer);
    }
    return *this;
}


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



Vector Vector::uminus() const
{
    auto kernel = get_kernel(Unary, "uminus");
    scalars::ScalarArray out_data(scalar_type(), dimension());
    devices::Buffer out_keys(m_key_buffer.clone());

    auto launch_params = get_kernel_launch_params();

    kernel(launch_params, out_data.mut_buffer(), m_scalar_buffer.buffer());

    return {p_basis, std::move(out_data), std::move(out_keys)};
}

Vector Vector::add(const Vector& other) const { return Vector(); }

Vector Vector::sub(const Vector& other) const { return Vector(); }

Vector Vector::left_smul(const scalars::Scalar& other) const
{
    const auto* stype = scalar_type();

    Vector result(p_basis, stype);

    if (!other.is_zero()) {
        auto kernel = get_kernel(Unary, "left_scalar_multiply");
        auto params = get_kernel_launch_params();

        result.m_scalar_buffer = scalars::ScalarArray(stype,
                                                      dimension());
        result.m_key_buffer = m_key_buffer.clone();

        kernel(params,
               result.m_scalar_buffer.mut_buffer(),
               m_scalar_buffer.buffer(),
               other
        );
    }

    return result;
}

Vector Vector::right_smul(const scalars::Scalar& other) const
{
    const auto* stype = scalar_type();
    Vector result(p_basis, stype);

    if (!other.is_zero()) {
        auto kernel = get_kernel(Unary, "right_scalar_multiply");
        auto params = get_kernel_launch_params();

        result.m_scalar_buffer = scalars::ScalarArray(stype, dimension());
        result.m_key_buffer = m_key_buffer.clone();

        kernel(params,
               result.m_scalar_buffer.mut_buffer(),
               m_scalar_buffer.buffer(),
               other);
    }

    return result;
}

Vector Vector::sdiv(const scalars::Scalar& other) const
{

    if (other.is_zero()) {
        throw std::domain_error("division by zero");
    }


    return Vector();
}

Vector& Vector::add_inplace(const Vector& other) { return *this; }

Vector& Vector::sub_inplace(const Vector& other) { return *this; }

Vector& Vector::smul_inplace(const scalars::Scalar& other)
{


    return *this;
}

Vector& Vector::sdiv_inplace(const scalars::Scalar& other) { return *this; }

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
    return os;
}
