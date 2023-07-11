#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_BASE_IMPL_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_BASE_IMPL_H_

#include "algebra_bundle.h"

#include <ostream>

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "algebra_base.h"
#include "context.h"

namespace rpy {
namespace algebra {

template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
AlgebraBundleBase<BundleInterface, DerivedImpl>::AlgebraBundleBase(
        const AlgebraBundleBase& other
)
{
    if (other.p_impl) { *this = other.p_impl->clone(); }
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
AlgebraBundleBase<BundleInterface, DerivedImpl>&
AlgebraBundleBase<BundleInterface, DerivedImpl>::operator=(
        const AlgebraBundleBase& other
)
{
    if (&other != this) {
        if (other.p_impl) { *this = other->clone(); }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t
AlgebraBundleBase<BundleInterface, DerivedImpl>::borrow() const
{
    RPY_CHECK(p_impl != nullptr);
    return p_impl->borrow();
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t
AlgebraBundleBase<BundleInterface, DerivedImpl>::borrow_mut()
{
    RPY_CHECK(p_impl != nullptr);
    return p_impl->borrow_mut();
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
dimn_t AlgebraBundleBase<BundleInterface, DerivedImpl>::dimension() const
{
    if (p_impl) { return p_impl->dimension(); }
    return 0;
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
dimn_t AlgebraBundleBase<BundleInterface, DerivedImpl>::size() const
{
    if (p_impl) { return p_impl->size(); }
    return 0;
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
bool AlgebraBundleBase<BundleInterface, DerivedImpl>::is_zero() const
{
    if (p_impl) { return p_impl->is_zero(); }
    return true;
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBundleBase<BundleInterface, DerivedImpl>::width() const
{
    if (p_impl) { return p_impl->width(); }
    return {};
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBundleBase<BundleInterface, DerivedImpl>::depth() const
{
    if (p_impl) { return p_impl->depth(); }
    return {};
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBundleBase<BundleInterface, DerivedImpl>::degree() const
{
    if (p_impl) { return p_impl->degree(); }
    return {};
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
VectorType
AlgebraBundleBase<BundleInterface, DerivedImpl>::storage_type() const noexcept
{
    if (p_impl) { return p_impl->storage_type(); }
    return VectorType::Sparse;
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
const scalars::ScalarType*
AlgebraBundleBase<BundleInterface, DerivedImpl>::coeff_type() const noexcept
{
    if (p_impl) { return p_impl->coeff_type(); }
    return nullptr;
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
scalars::Scalar
AlgebraBundleBase<BundleInterface, DerivedImpl>::operator[](key_type k) const
{
    if (p_impl) { return p_impl->get(k); }
    return scalars::Scalar();
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
scalars::Scalar
AlgebraBundleBase<BundleInterface, DerivedImpl>::operator[](key_type k)
{
    if (p_impl) { return p_impl->get_mut(k); }
    return scalars::Scalar();
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t
AlgebraBundleBase<BundleInterface, DerivedImpl>::uminus() const
{
    if (p_impl) { return p_impl->uminus(); }
    return {};
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t
AlgebraBundleBase<BundleInterface, DerivedImpl>::add(const algebra_t& rhs) const
{
    if (!rhs.p_impl) {
        if (!p_impl) { return algebra_t(); }
        return p_impl->clone();
    }
    if (!p_impl) { return rhs->clone(); }
    return p_impl->add(rhs);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t
AlgebraBundleBase<BundleInterface, DerivedImpl>::sub(const algebra_t& rhs) const
{
    if (!rhs.p_impl) {
        if (!p_impl) { return algebra_t(); }
        return p_impl->clone();
    }
    if (!p_impl) { return rhs->uminus(); }
    return p_impl->sub(rhs);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t
AlgebraBundleBase<BundleInterface, DerivedImpl>::mul(const algebra_t& rhs) const
{
    if (!p_impl || !rhs.p_impl) { return algebra_t(); }
    return p_impl->mul(rhs);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t
AlgebraBundleBase<BundleInterface, DerivedImpl>::smul(const scalars::Scalar& rhs
) const
{
    if (!p_impl) { return algebra_t(); }
    return p_impl->smul(rhs);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t
AlgebraBundleBase<BundleInterface, DerivedImpl>::sdiv(const scalars::Scalar& rhs
) const
{
    if (!p_impl) { return algebra_t(); }
    return p_impl->sdiv(rhs);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::add_inplace(
        const algebra_t& rhs
)
{
    if (rhs.p_impl) {
        if (!p_impl) {
            *this = rhs->clone();
        } else {
            p_impl->add_inplace(rhs);
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::sub_inplace(
        const algebra_t& rhs
)
{
    if (rhs.p_impl) {
        if (!p_impl) {
            *this = rhs->uminus();
        } else {
            p_impl->sub_inplace(rhs);
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::mul_inplace(
        const algebra_t& rhs
)
{
    if (p_impl) {
        if (rhs.p_impl) {
            p_impl->mul_inplace(rhs);
        } else {
            p_impl->clear();
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::smul_inplace(
        const scalars::Scalar& rhs
)
{
    if (p_impl) { p_impl->smul_inplace(rhs); }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::sdiv_inplace(
        const scalars::Scalar& rhs
)
{
    if (p_impl) { p_impl->sdiv_inplace(rhs); }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::add_scal_mul(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (lhs.p_impl) {
        if (p_impl) {
            p_impl->add_scal_mul(lhs, rhs);
        } else {
            *this = lhs->smul(rhs);
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::sub_scal_mul(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (lhs.p_impl) {
        if (p_impl) {
            p_impl->sub_scal_mul(lhs, rhs);
        } else {
            *this = lhs->smul(-rhs);
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::add_scal_div(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (lhs.p_impl) {
        if (p_impl) {
            p_impl->add_scal_div(lhs, rhs);
        } else {
            *this = lhs->sdiv(rhs);
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::sub_scal_div(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (lhs.p_impl) {
        if (p_impl) {
            p_impl->sub_scal_div(lhs, rhs);
        } else {
            *this = lhs->sdiv(-rhs);
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::add_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    if (lhs.p_impl && rhs.p_impl) {
        if (p_impl) {
            p_impl->add_mul(lhs, rhs);
        } else {
            *this = lhs->mul(rhs);
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::sub_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    if (lhs.p_impl && rhs.p_impl) {
        if (p_impl) {
            p_impl->sub_mul(lhs, rhs);
        } else {
            *this = (lhs->mul(rhs))->uminus();
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::mul_smul(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (p_impl) {
        if (lhs.p_impl) {
            p_impl->mul_smul(lhs, rhs);
        } else {
            p_impl->clear();
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t&
AlgebraBundleBase<BundleInterface, DerivedImpl>::mul_sdiv(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    if (p_impl) {
        if (lhs.p_impl) {
            p_impl->mul_sdiv(lhs, rhs);
        } else {
            p_impl->clear();
        }
    }
    return downcast(*this);
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
std::ostream&
AlgebraBundleBase<BundleInterface, DerivedImpl>::print(std::ostream& os) const
{
    if (p_impl) {
        p_impl->print(os);
    } else {
        dtl::print_empty_algebra(os);
    }
    return os;
}
template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl>
bool AlgebraBundleBase<BundleInterface, DerivedImpl>::operator==(
        const algebra_t& other
) const
{
    if (p_impl && other.p_impl) {
        return p_impl->equals(other);
    } else if (p_impl) {
        return p_impl->is_zero();
    } else if (other.p_impl) {
        return other->is_zero();
    }
    return true;
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_BASE_IMPL_H_
