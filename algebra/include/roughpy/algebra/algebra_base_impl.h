#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BASE_IMPL_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BASE_IMPL_H_

#include "algebra_base.h"

#include <ostream>

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include "context.h"

namespace rpy {
namespace algebra {

#define RPY_CHECK_CONTEXTS(OTHER)                                              \
    dtl::check_contexts_compatible(context(), (OTHER).context())

template <typename Algebra, typename BasisType>
typename dtl::AlgebraBasicProperties<Algebra, BasisType>::id_t
dtl::AlgebraBasicProperties<Algebra, BasisType>::id() const noexcept
{
    return 0;
}
template <typename Algebra, typename BasisType>
optional<deg_t> dtl::AlgebraBasicProperties<Algebra, BasisType>::degree() const
{
    return optional<deg_t>();
}
template <typename Algebra, typename BasisType>
optional<deg_t> dtl::AlgebraBasicProperties<Algebra, BasisType>::width() const
{
    return optional<deg_t>();
}
template <typename Algebra, typename BasisType>
optional<deg_t> dtl::AlgebraBasicProperties<Algebra, BasisType>::depth() const
{
    return optional<deg_t>();
}

template <typename Base>
optional<scalars::ScalarArray>
dtl::AlgebraIteratorMethods<Base>::dense_data() const
{
    return {};
}

template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t
dtl::AlgebraArithmetic<Base>::add(const algebra_t& other) const
{
    auto result = this->clone();
    result->add_inplace(other);
    return result;
}
template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t
dtl::AlgebraArithmetic<Base>::sub(const algebra_t& other) const
{
    auto result = this->clone();
    result->sub_inplace(other);
    return result;
}
template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t
dtl::AlgebraArithmetic<Base>::mul(const algebra_t& other) const
{
    auto result = this->clone();
    result->mul_inplace(other);
    return result;
}
template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t
dtl::AlgebraArithmetic<Base>::smul(const scalars::Scalar& other) const
{
    auto result = this->clone();
    result->smul_inplace(other);
    return result;
}
template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t
dtl::AlgebraArithmetic<Base>::sdiv(const scalars::Scalar& other) const
{
    auto result = this->clone();
    result->sdiv_inplace(other);
    return result;
}

template <typename Base>
void dtl::AlgebraArithmetic<Base>::add_scal_mul(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    auto tmp = rhs.smul(scalar);
    add_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::sub_scal_mul(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    auto tmp = rhs.smul(scalar);
    sub_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::add_scal_div(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    auto tmp = rhs.sdiv(scalar);
    add_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::sub_scal_div(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    auto tmp = rhs.sdiv(scalar);
    sub_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::add_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    auto tmp = lhs.mul(rhs);
    add_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::sub_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    auto tmp = lhs.mul(rhs);
    sub_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::mul_smul(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    auto tmp = rhs.smul(scalar);
    mul_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::mul_sdiv(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    auto tmp = rhs.sdiv(scalar);
    mul_inplace(tmp);
}

template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>::AlgebraBase(context_pointer ctx)
    : p_impl(nullptr)
{
    /*
     * Try and create a new empty instance by appealing to the context
     * and passing in the type to be created. This will return either
     * a pointer to a AlgebraInterfaceTag, which will necessarily actually
     * point to an object of type Interface, or a null pointer if this
     * construction is not possible.
     */

    p_impl = dtl::downcast_interface_ptr<Interface>(
            dtl::try_create_new_empty(std::move(ctx), algebra_t::s_alg_type)
    );
}

template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>::AlgebraBase(const AlgebraBase& other)
{
    if (other.p_impl) { *this = other.p_impl->clone(); }
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>::AlgebraBase(AlgebraBase&& other) noexcept
    : p_impl(std::move(other.p_impl))
{}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>&
AlgebraBase<Interface, DerivedImpl>::operator=(const AlgebraBase& other)
{
    if (&other != this) {
        if (other.p_impl) {
            *this = other.p_impl->clone();
        } else {
            p_impl.reset();
        }
    }
    return *this;
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>&
AlgebraBase<Interface, DerivedImpl>::operator=(AlgebraBase&& other) noexcept
{
    if (&other != this) { p_impl = std::move(other.p_impl); }
    return *this;
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
context_pointer AlgebraBase<Interface, DerivedImpl>::context() const noexcept
{
    return (p_impl) ? p_impl->context() : nullptr;
}

template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t
AlgebraBase<Interface, DerivedImpl>::borrow() const
{
    if (p_impl) { return p_impl->borrow(); }
    return algebra_t();
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::borrow_mut()
{
    if (p_impl) { return p_impl->borrow_mut(); }
    return algebra_t();
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t
AlgebraBase<Interface, DerivedImpl>::add(const algebra_t& rhs) const
{
    if (is_equivalent_to_zero(rhs)) {
        if (is_equivalent_to_zero(*this)) { return algebra_t(); }
        return p_impl->clone();
    }
    RPY_CHECK_CONTEXTS(rhs);

    if (is_equivalent_to_zero(*this)) { return rhs.p_impl->clone(); }
    return p_impl->add(rhs);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t
AlgebraBase<Interface, DerivedImpl>::sub(const algebra_t& rhs) const
{
    if (is_equivalent_to_zero(rhs)) {
        if (is_equivalent_to_zero(*this)) { return algebra_t(); }
        return p_impl->clone();
    }

    RPY_CHECK_CONTEXTS(rhs);

    if (is_equivalent_to_zero(*this)) { return rhs.p_impl->uminus(); }
    return p_impl->sub(rhs);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t
AlgebraBase<Interface, DerivedImpl>::mul(const algebra_t& rhs) const
{
    if (is_equivalent_to_zero(rhs)) {
        if (is_equivalent_to_zero(*this)) { return algebra_t(); }
        return p_impl->clone();
    }

    RPY_CHECK_CONTEXTS(rhs);

    if (is_equivalent_to_zero(*this)) { return rhs.p_impl->clone(); }
    return p_impl->mul(rhs);
}

template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
dimn_t AlgebraBase<Interface, DerivedImpl>::dimension() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->dimension(); }
    return 0;
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
dimn_t AlgebraBase<Interface, DerivedImpl>::size() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->size(); }
    return 0;
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
bool AlgebraBase<Interface, DerivedImpl>::is_zero() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->is_zero(); }
    return true;
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBase<Interface, DerivedImpl>::width() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->width(); }
    return {};
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBase<Interface, DerivedImpl>::depth() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->depth(); }
    return {};
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBase<Interface, DerivedImpl>::degree() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->degree(); }
    return {};
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
VectorType AlgebraBase<Interface, DerivedImpl>::storage_type() const noexcept
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->storage_type(); }
    return VectorType::Sparse;
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
const scalars::ScalarType*
AlgebraBase<Interface, DerivedImpl>::coeff_type() const noexcept
{
    if (p_impl) { return p_impl->coeff_type(); }
    return nullptr;
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
scalars::Scalar AlgebraBase<Interface, DerivedImpl>::operator[](key_type k
) const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->get(k); }
    return {};
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
scalars::Scalar AlgebraBase<Interface, DerivedImpl>::operator[](key_type k)
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->get_mut(k); }
    return {};
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBase<Interface, DerivedImpl>::const_iterator
AlgebraBase<Interface, DerivedImpl>::begin() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->begin(); }
    return {};
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBase<Interface, DerivedImpl>::const_iterator
AlgebraBase<Interface, DerivedImpl>::end() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->end(); }
    return {};
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
optional<rpy::scalars::ScalarArray>
AlgebraBase<Interface, DerivedImpl>::dense_data() const
{
    if (!is_equivalent_to_zero(*this)) { return p_impl->dense_data(); }
    return {};
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t
AlgebraBase<Interface, DerivedImpl>::uminus() const
{
    if (is_equivalent_to_zero(*this)) { return algebra_t(); }
    return p_impl->uminus();
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t
AlgebraBase<Interface, DerivedImpl>::smul(const scalars::Scalar& rhs) const
{
    if (is_equivalent_to_zero(*this)) { return algebra_t(); }
    if (rhs.is_zero()) { return p_impl->zero_like(); }
    // The implementation should perform the necessary scalar casting
    return p_impl->smul(rhs);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t
AlgebraBase<Interface, DerivedImpl>::sdiv(const scalars::Scalar& rhs) const
{
    if (is_equivalent_to_zero(*this)) { return algebra_t(); }
    if (rhs.is_zero()) { RPY_THROW(std::invalid_argument, "cannot divide by zero"); }
    // The implementation should perform the necessary scalar casting
    return p_impl->sdiv(rhs);
}

template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::add_inplace(const algebra_t& rhs)
{
    if (is_equivalent_to_zero(rhs)) { return downcast(*this); }
    if (is_equivalent_to_zero(*this)) { *this = algebra_t(); }
    RPY_CHECK_CONTEXTS(rhs);
    p_impl->add_inplace(rhs);
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::sub_inplace(const algebra_t& rhs)
{
    if (is_equivalent_to_zero(rhs)) { return downcast(*this); }
    if (is_equivalent_to_zero(*this)) { *this = algebra_t(); }
    RPY_CHECK_CONTEXTS(rhs);
    p_impl->sub_inplace(rhs);
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::mul_inplace(const algebra_t& rhs)
{
    if (is_equivalent_to_zero(rhs)) { return downcast(*this); }
    if (is_equivalent_to_zero(*this)) { *this = algebra_t(); }
    RPY_CHECK_CONTEXTS(rhs);
    p_impl->mul_inplace(rhs);
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::smul_inplace(const scalars::Scalar& rhs)
{
    if (!is_equivalent_to_zero(*this)) { p_impl->smul_inplace(rhs); }
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::sdiv_inplace(const scalars::Scalar& rhs)
{
    if (!is_equivalent_to_zero(*this)) {
        if (rhs.is_zero()) {
            RPY_THROW(std::invalid_argument, "cannot divide by zero");
        }
        p_impl->sdiv_inplace(rhs);
    }
    return downcast(*this);
}

template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::add_scal_mul(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (!is_equivalent_to_zero(lhs) && rhs.is_zero()) {
        RPY_CHECK_CONTEXTS(lhs);

        if (!is_equivalent_to_zero(*this)) {
            p_impl->add_scal_mul(lhs, rhs);
        } else {
            *this = lhs.smul(rhs);
        }
    }
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::sub_scal_mul(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (!is_equivalent_to_zero(lhs) && rhs.is_zero()) {
        RPY_CHECK_CONTEXTS(lhs);

        if (!is_equivalent_to_zero(*this)) {
            p_impl->sub_scal_mul(lhs, rhs);
        } else {
            *this = lhs.smul(rhs).uminus();
        }
    }
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::add_scal_div(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (!is_equivalent_to_zero(lhs)) {
        RPY_CHECK_CONTEXTS(lhs);

        if (rhs.is_zero()) {
            RPY_THROW(std::invalid_argument, "cannot divide by zero");
        }
        if (!is_equivalent_to_zero(*this)) {
            p_impl->add_scal_div(lhs, rhs);
        } else {
            *this = lhs.sdiv(rhs);
        }
    }
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t&
AlgebraBase<Interface, DerivedImpl>::sub_scal_div(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (!is_equivalent_to_zero(lhs)) {
        RPY_CHECK_CONTEXTS(lhs);

        if (rhs.is_zero()) {
            RPY_THROW(std::invalid_argument, "cannot divide by zero");
        }
        if (!is_equivalent_to_zero(*this)) {
            p_impl->sub_scal_div(lhs, rhs);
        } else {
            *this = lhs.sdiv(rhs).uminus();
        }
    }
    return downcast(*this);
}

template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t& AlgebraBase<Interface, DerivedImpl>::add_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    if (!is_equivalent_to_zero(lhs) && !is_equivalent_to_zero(rhs)) {
        RPY_CHECK_CONTEXTS(lhs);
        RPY_CHECK_CONTEXTS(rhs);

        if (is_equivalent_to_zero(*this)) {
            *this = lhs.mul(rhs);
        } else {
            p_impl->add_mul(lhs, rhs);
        }
    }
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t& AlgebraBase<Interface, DerivedImpl>::sub_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    if (!is_equivalent_to_zero(lhs) && !is_equivalent_to_zero(rhs)) {
        RPY_CHECK_CONTEXTS(lhs);
        RPY_CHECK_CONTEXTS(rhs);

        if (is_equivalent_to_zero(*this)) {
            *this = lhs.mul(rhs).uminus();
        } else {
            p_impl->sub_mul(lhs, rhs);
        }
    }
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t& AlgebraBase<Interface, DerivedImpl>::mul_smul(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (!is_equivalent_to_zero(lhs) && !rhs.is_zero()) {
        RPY_CHECK_CONTEXTS(lhs);

        if (!is_equivalent_to_zero(*this)) { p_impl->mul_smul(lhs, rhs); }
    } else if (!is_equivalent_to_zero(*this)) {
        p_impl->clear();
    }
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t& AlgebraBase<Interface, DerivedImpl>::mul_sdiv(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (!is_equivalent_to_zero(lhs)) {
        RPY_CHECK_CONTEXTS(lhs);

        if (rhs.is_zero()) {
            RPY_THROW(std::invalid_argument, "cannot divide by zero");
        }
        if (!is_equivalent_to_zero(*this)) { p_impl->mul_sdiv(lhs, rhs); }
    } else if (!is_equivalent_to_zero(*this)) {
        p_impl->clear();
    }
    return downcast(*this);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
bool AlgebraBase<Interface, DerivedImpl>::operator==(const algebra_t& other
) const
{
    if (is_equivalent_to_zero(*this)) {
        return is_equivalent_to_zero(other) || other->is_zero();
    }
    if (is_equivalent_to_zero(other)) { return p_impl->is_zero(); }

    if (!context()->check_compatible(*other.context())) { return false; }

    return p_impl->equals(other);
}
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
std::ostream& AlgebraBase<Interface, DerivedImpl>::print(std::ostream& os) const
{
    if (is_equivalent_to_zero(*this)) {
        dtl::print_empty_algebra(os);
    } else {
        p_impl->print(os);
    }
    return os;
}

template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
inline std::ostream&
operator<<(std::ostream& os, const AlgebraBase<Interface, DerivedImpl>& alg)
{
    return alg.print(os);
}

#undef RPY_CHECK_CONTEXTS
template <
        typename Interface,
        template <typename, template <typename> class> class Derived>
template <typename Archive>
void AlgebraBase<Interface, Derived>::save(
        Archive& archive, const std::uint32_t RPY_UNUSED_VAR version
) const
{
    context_pointer ctx = (p_impl) ? p_impl->context() : nullptr;
    auto spec = get_context_spec(ctx);
    RPY_SERIAL_SERIALIZE_NVP("width", spec.width);
    RPY_SERIAL_SERIALIZE_NVP("depth", spec.depth);
    RPY_SERIAL_SERIALIZE_NVP("scalar_type_id", spec.stype_id);
    RPY_SERIAL_SERIALIZE_NVP("backend", spec.backend);
    RPY_SERIAL_SERIALIZE_NVP("algebra_type", algebra_t::s_alg_type);

    auto stype = storage_type();
    RPY_SERIAL_SERIALIZE_NVP("storage_type", stype);
    RPY_SERIAL_SERIALIZE_NVP("has_values", static_cast<bool>(p_impl));

    if (p_impl) {
        if (stype == VectorType::Dense) {
            auto data = *this->dense_data();
            RPY_SERIAL_SERIALIZE_NVP("dense_data", data);
        } else {
            auto sz = this->size();
            RPY_SERIAL_SERIALIZE_SIZE(sz);

            for (auto&& item : *this) {
                RPY_SERIAL_SERIALIZE_BARE(
                        std::make_pair(item.key(), item.value())
                );
            }
        }
    }
}

template <
        typename Interface,
        template <typename, template <typename> class> class Derived>
template <typename Archive>
void AlgebraBase<Interface, Derived>::load(
        Archive& archive, const std::uint32_t RPY_UNUSED_VAR version
)
{
    BasicContextSpec spec;
    RPY_SERIAL_SERIALIZE_NVP("width", spec.width);
    RPY_SERIAL_SERIALIZE_NVP("depth", spec.depth);
    RPY_SERIAL_SERIALIZE_NVP("scalar_type_id", spec.stype_id);
    RPY_SERIAL_SERIALIZE_NVP("backend", spec.backend);

    auto ctx = from_context_spec(spec);

    AlgebraType atype;
    RPY_SERIAL_SERIALIZE_NVP("algebra_type", atype);

    VectorType vtype;
    RPY_SERIAL_SERIALIZE_NVP("storage_type", vtype);

    bool has_values;
    RPY_SERIAL_SERIALIZE_VAL(has_values);

    if (!has_values) { return; }

    if (vtype == VectorType::Dense) {

        scalars::ScalarArray tmp;
        RPY_SERIAL_SERIALIZE_NVP("dense_data", tmp);
        p_impl = dtl::downcast_interface_ptr<Interface>(
                dtl::construct_dense_algebra(std::move(tmp), ctx, atype)
        );
    } else {
        p_impl = dtl::downcast_interface_ptr<Interface>(
                dtl::try_create_new_empty(ctx, atype)
        );

        serial::size_type size;
        RPY_SERIAL_SERIALIZE_SIZE(size);

        for (serial::size_type i = 0; i < size; ++i) {
            std::pair<typename Interface::key_type, scalars::Scalar> val;
            RPY_SERIAL_SERIALIZE_BARE(val);
            p_impl->get_mut(val.first) = val.second;
        }
    }
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_BASE_IMPL_H_
