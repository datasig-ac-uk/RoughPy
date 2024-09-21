//
// Created by sam on 16/02/24.
//

#include "free_tensor.h"

#include "multiplication/FreeTensorMultiplication.h"

using namespace rpy;
using namespace rpy::algebra;

FreeTensor FreeTensor::new_like(const FreeTensor& arg) noexcept { return {}; }
FreeTensor FreeTensor::clone(const FreeTensor& arg) noexcept { return {}; }
FreeTensor FreeTensor::from(Vector&& arg) noexcept { return {}; }

FreeTensor FreeTensor::unit_like(const FreeTensor& arg) noexcept { return {}; }
FreeTensor::FreeTensor() : AlgebraBase(Vector()) {}
FreeTensor&
FreeTensor::fma(const Vector& lhs, const Vector& rhs, const ops::Operator& op)
{
    return fma(lhs, rhs, op, basis()->max_degree());
}
FreeTensor& FreeTensor::fma(
        const Vector& lhs,
        const Vector& rhs,
        const ops::Operator& op,
        deg_t max_degree,
        deg_t lhs_min_deg,
        deg_t rhs_min_deg
)
{
    const auto basis = this->basis();
    const FreeTensorMultiplication ftm(
            basis->alphabet_size(),
            basis->max_degree()
    );

    ftm.eval(
            this->as_vector(),
            lhs,
            rhs,
            op,
            max_degree,
            lhs_min_deg,
            rhs_min_deg
    );

    return *this;
}
FreeTensor&
FreeTensor::multiply_inplace(const Vector& rhs, const ops::Operator& op)
{
    return multiply_inplace(rhs, op, basis()->max_degree());
}
FreeTensor& FreeTensor::multiply_inplace(
        const Vector& rhs,
        const ops::Operator& op,
        deg_t max_degree,
        deg_t lhs_min_deg,
        deg_t rhs_min_deg
)
{
    const auto basis = this->basis();
    const FreeTensorMultiplication ftm(
            basis->alphabet_size(),
            basis->max_degree()
    );

    ftm.eval_inplace(
            this->as_vector(),
            rhs,
            op,
            max_degree,
            lhs_min_deg,
            rhs_min_deg
    );
    return *this;
}

FreeTensor FreeTensor::exp() const
{
    return fused_multiply_exp(unit_like(*this));
}
FreeTensor FreeTensor::log() const
{
    auto result = new_like(*this);
    const auto max_degree = basis()->max_degree();
    const auto unit = unit_like(*this);
    const auto stype = scalar_type();

    result.as_vector().resize_base_dim(basis()->max_dimension());

    const ops::IdentityOperator op{};// NOLINT(*-identifier-length)
    for (deg_t degree = max_degree; degree >= 1; --degree) {
        auto divisor = devices::math::from_rational(stype, 1, degree);
        if (degree % 2 == 0) {
            result.as_vector().sub_scal_mul(unit.as_vector(), divisor);
        } else {
            result.as_vector().add_scal_mul(unit.as_vector(), divisor);
        }

        result.multiply_inplace(as_vector(), op, max_degree, 0, 1);
    }
    return result;
}

FreeTensor FreeTensor::antipode() const
{
    auto result = new_like(*this);

    const auto basis = this->basis();
    const FreeTensorMultiplication ftm(
            basis->alphabet_size(),
            basis->max_degree()
    );

    ftm.antipode(result.as_vector(), as_vector());
    return result;
}

FreeTensor FreeTensor::fused_multiply_exp(const FreeTensor& other) const
{
    auto result = new_like(*this);
    const auto max_degree = basis()->max_degree();

    result.as_vector().resize_base_dim(basis()->max_dimension());

    auto divisor = devices::math::reciprocal(Scalar(scalar_type(), max_degree));
    const ops::RightMultiplyOperator op(divisor);
    for (deg_t degree = max_degree; degree >= 1; --degree) {
        result.multiply_inplace(as_vector(), op, max_degree - degree + 1, 0, 1);
        result += other;
        divisor -= 1;
    }
    return result;
}

FreeTensor& FreeTensor::fused_multiply_exp_inplace(const FreeTensor& other)
{
    auto tmp = fused_multiply_exp(other);
    std::swap(*this, tmp);
    return *this;
}
