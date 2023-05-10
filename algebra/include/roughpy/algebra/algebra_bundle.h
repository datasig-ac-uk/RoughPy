// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_

#include "algebra_base.h"
#include "algebra_fwd.h"

#include <roughpy/platform/serialization.h>

namespace rpy {
namespace algebra {

template <typename Bundle, typename Base, typename Fibre>
class ROUGHPY_ALGEBRA_EXPORT BundleInterface
    : public dtl::algebra_base_resolution<Bundle, typename Base::basis_type,
                                          dtl::AlgebraArithmetic,
                                          dtl::AlgebraElementAccess>::type {
public:
    using base_alg_t = Base;
    using fibre_alg_t = Fibre;
    using base_interface_t = typename Base::interface_t;
    using fibre_interface_t = typename Fibre::interface_t;

    virtual base_alg_t base() = 0;
    virtual fibre_alg_t fibre() = 0;
};

template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl = dtl::with_interface<BundleInterface>::template type>
class AlgebraBundleBase {

    explicit AlgebraBundleBase(std::unique_ptr<BundleInterface> &&impl) : p_impl(std::move(impl)) {}

    explicit AlgebraBundleBase(BundleInterface *impl) : p_impl(impl) {}

protected:
    std::unique_ptr<BundleInterface> p_impl;

public:
    using interface_t = BundleInterface;
    using base_alg_t = typename BundleInterface::base_alg_t;
    using fibre_alg_t = typename BundleInterface::fibre_alg_t;

    using basis_type = typename base_alg_t::basis_type;
    using key_type = typename basis_type::key_type;
    using fibre_basis_type = typename fibre_alg_t::basis_type;

    using algebra_t = typename BundleInterface::algebra_t;

    AlgebraBundleBase() : p_impl(nullptr) {}

    AlgebraBundleBase(const AlgebraBundleBase &other);
    AlgebraBundleBase(AlgebraBundleBase &&other) noexcept = default;

    AlgebraBundleBase &operator=(const AlgebraBundleBase &other);
    AlgebraBundleBase &operator=(AlgebraBundleBase &&other) noexcept = default;

    algebra_t borrow() const;
    algebra_t borrow_mut();

    const BundleInterface &operator*() const noexcept { return *p_impl; }
    BundleInterface &operator*() noexcept { return *p_impl; }
    const BundleInterface *operator->() const noexcept { return p_impl.get(); }
    BundleInterface *operator->() noexcept { return p_impl.get(); }

    constexpr operator bool() const noexcept { return static_cast<bool>(p_impl); }

    dimn_t dimension() const;
    dimn_t size() const;
    bool is_zero() const;
    optional<deg_t> width() const;
    optional<deg_t> depth() const;
    optional<deg_t> degree() const;

    VectorType storage_type() const noexcept;
    const scalars::ScalarType *coeff_type() const noexcept;

    scalars::Scalar operator[](key_type k) const;
    scalars::Scalar operator[](key_type k);

protected:
    static algebra_t &downcast(AlgebraBundleBase &arg) { return static_cast<algebra_t &>(arg); }
    static const algebra_t &downcast(const AlgebraBundleBase &arg) { return static_cast<const algebra_t &>(arg); }

public:
    algebra_t uminus() const;
    algebra_t add(const algebra_t &rhs) const;
    algebra_t sub(const algebra_t &rhs) const;
    algebra_t mul(const algebra_t &rhs) const;
    algebra_t smul(const scalars::Scalar &rhs) const;
    algebra_t sdiv(const scalars::Scalar &rhs) const;

    algebra_t &add_inplace(const algebra_t &rhs);
    algebra_t &sub_inplace(const algebra_t &rhs);
    algebra_t &mul_inplace(const algebra_t &rhs);
    algebra_t &smul_inplace(const scalars::Scalar &rhs);
    algebra_t &sdiv_inplace(const scalars::Scalar &rhs);

    algebra_t &add_scal_mul(const algebra_t &lhs, const scalars::Scalar &rhs);
    algebra_t &sub_scal_mul(const algebra_t &lhs, const scalars::Scalar &rhs);
    algebra_t &add_scal_div(const algebra_t &lhs, const scalars::Scalar &rhs);
    algebra_t &sub_scal_div(const algebra_t &lhs, const scalars::Scalar &rhs);

    algebra_t &add_mul(const algebra_t &lhs, const algebra_t &rhs);
    algebra_t &sub_mul(const algebra_t &lhs, const algebra_t &rhs);
    algebra_t &mul_smul(const algebra_t &lhs, const scalars::Scalar &rhs);
    algebra_t &mul_sdiv(const algebra_t &lhs, const scalars::Scalar &rhs);

    std::ostream &print(std::ostream &os) const;

    bool operator==(const algebra_t &other) const;
    bool operator!=(const algebra_t &other) const { return !operator==(other); }

#ifndef RPY_DISABLE_SERIALIZATION
private:
    friend rpy::serialization_access;

    RPY_SERIAL_SPLIT_MEMBER();

    template <typename Ar>
    void save(Ar &ar, const unsigned int /*version*/) const {
        context_pointer ctx = (p_impl) ? p_impl->context() : nullptr;
        auto spec = get_context_spec(ctx);
        ar << spec.width;
        ar << spec.depth;
        ar << spec.stype_id;
        ar << spec.backend;
        ar << algebra_t::s_alg_type;
        ar << alg_to_raw_bytes(ctx, algebra_t::s_alg_type, p_impl.get());
    }

    template <typename Ar>
    void load(Ar &ar, const unsigned int /*version*/) {
        BasicContextSpec spec;
        ar >> spec.width;
        ar >> spec.depth;
        ar >> spec.stype_id;
        ar >> spec.backend;

        auto ctx = from_context_spec(spec);

        AlgebraType atype;
        ar >> atype;
        std::vector<byte> raw_data;
        ar >> raw_data;
        UnspecifiedAlgebraType alg = alg_from_raw_bytes(ctx, atype, raw_data);

        RPY_CHECK(algebra_t::s_alg_type == atype);
        p_impl = std::unique_ptr<BundleInterface>(reinterpret_cast<BundleInterface *>(alg /*.release()*/));
    }

#endif
};

template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBundleBase<BundleInterface, DerivedImpl>::AlgebraBundleBase(const AlgebraBundleBase &other) {
    if (other.p_impl) {
        *this = other.p_impl->clone();
    }
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBundleBase<BundleInterface, DerivedImpl> &AlgebraBundleBase<BundleInterface, DerivedImpl>::operator=(const AlgebraBundleBase &other) {
    if (&other != this) {
        if (other.p_impl) {
            *this = other->clone();
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t AlgebraBundleBase<BundleInterface, DerivedImpl>::borrow() const {
    RPY_CHECK(p_impl != nullptr);
    return p_impl->borrow();
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t AlgebraBundleBase<BundleInterface, DerivedImpl>::borrow_mut() {
    RPY_CHECK(p_impl != nullptr);
    return p_impl->borrow_mut();
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
dimn_t AlgebraBundleBase<BundleInterface, DerivedImpl>::dimension() const {
    if (p_impl) {
        return p_impl->dimension();
    }
    return 0;
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
dimn_t AlgebraBundleBase<BundleInterface, DerivedImpl>::size() const {
    if (p_impl) {
        return p_impl->size();
    }
    return 0;
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
bool AlgebraBundleBase<BundleInterface, DerivedImpl>::is_zero() const {
    if (p_impl) {
        return p_impl->is_zero();
    }
    return true;
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBundleBase<BundleInterface, DerivedImpl>::width() const {
    if (p_impl) {
        return p_impl->width();
    }
    return {};
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBundleBase<BundleInterface, DerivedImpl>::depth() const {
    if (p_impl) {
        return p_impl->depth();
    }
    return {};
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBundleBase<BundleInterface, DerivedImpl>::degree() const {
    if (p_impl) {
        return p_impl->degree();
    }
    return {};
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
VectorType AlgebraBundleBase<BundleInterface, DerivedImpl>::storage_type() const noexcept {
    if (p_impl) {
        return p_impl->storage_type();
    }
    return VectorType::Sparse;
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
const scalars::ScalarType *AlgebraBundleBase<BundleInterface, DerivedImpl>::coeff_type() const noexcept {
    if (p_impl) {
        return p_impl->coeff_type();
    }
    return nullptr;
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
scalars::Scalar AlgebraBundleBase<BundleInterface, DerivedImpl>::operator[](key_type k) const {
    if (p_impl) {
        return p_impl->get(k);
    }
    return scalars::Scalar();
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
scalars::Scalar AlgebraBundleBase<BundleInterface, DerivedImpl>::operator[](key_type k) {
    if (p_impl) {
        return p_impl->get_mut(k);
    }
    return scalars::Scalar();
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t AlgebraBundleBase<BundleInterface, DerivedImpl>::uminus() const {
    if (p_impl) {
        return p_impl->uminus();
    }
    return {};
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t AlgebraBundleBase<BundleInterface, DerivedImpl>::add(const algebra_t &rhs) const {
    if (!rhs.p_impl) {
        if (!p_impl) {
            return algebra_t();
        }
        return p_impl->clone();
    }
    if (!p_impl) {
        return rhs->clone();
    }
    return p_impl->add(rhs);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t AlgebraBundleBase<BundleInterface, DerivedImpl>::sub(const algebra_t &rhs) const {
    if (!rhs.p_impl) {
        if (!p_impl) {
            return algebra_t();
        }
        return p_impl->clone();
    }
    if (!p_impl) {
        return rhs->uminus();
    }
    return p_impl->sub(rhs);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t AlgebraBundleBase<BundleInterface, DerivedImpl>::mul(const algebra_t &rhs) const {
    if (!p_impl || !rhs.p_impl) {
        return algebra_t();
    }
    return p_impl->mul(rhs);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t AlgebraBundleBase<BundleInterface, DerivedImpl>::smul(const scalars::Scalar &rhs) const {
    if (!p_impl) {
        return algebra_t();
    }
    return p_impl->smul(rhs);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t AlgebraBundleBase<BundleInterface, DerivedImpl>::sdiv(const scalars::Scalar &rhs) const {
    if (!p_impl) {
        return algebra_t();
    }
    return p_impl->sdiv(rhs);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::add_inplace(const algebra_t &rhs) {
    if (rhs.p_impl) {
        if (!p_impl) {
            *this = rhs->clone();
        } else {
            p_impl->add_inplace(rhs);
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::sub_inplace(const algebra_t &rhs) {
    if (rhs.p_impl) {
        if (!p_impl) {
            *this = rhs->uminus();
        } else {
            p_impl->sub_inplace(rhs);
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::mul_inplace(const algebra_t &rhs) {
    if (p_impl) {
        if (rhs.p_impl) {
            p_impl->mul_inplace(rhs);
        } else {
            p_impl->clear();
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::smul_inplace(const scalars::Scalar &rhs) {
    if (p_impl) {
        p_impl->smul_inplace(rhs);
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::sdiv_inplace(const scalars::Scalar &rhs) {
    if (p_impl) {
        p_impl->sdiv_inplace(rhs);
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::add_scal_mul(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (lhs.p_impl) {
        if (p_impl) {
            p_impl->add_scal_mul(lhs, rhs);
        } else {
            *this = lhs->smul(rhs);
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::sub_scal_mul(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (lhs.p_impl) {
        if (p_impl) {
            p_impl->sub_scal_mul(lhs, rhs);
        } else {
            *this = lhs->smul(-rhs);
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::add_scal_div(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (lhs.p_impl) {
        if (p_impl) {
            p_impl->add_scal_div(lhs, rhs);
        } else {
            *this = lhs->sdiv(rhs);
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::sub_scal_div(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (lhs.p_impl) {
        if (p_impl) {
            p_impl->sub_scal_div(lhs, rhs);
        } else {
            *this = lhs->sdiv(-rhs);
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::add_mul(const algebra_t &lhs, const algebra_t &rhs) {
    if (lhs.p_impl && rhs.p_impl) {
        if (p_impl) {
            p_impl->add_mul(lhs, rhs);
        } else {
            *this = lhs->mul(rhs);
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::sub_mul(const algebra_t &lhs, const algebra_t &rhs) {
    if (lhs.p_impl && rhs.p_impl) {
        if (p_impl) {
            p_impl->sub_mul(lhs, rhs);
        } else {
            *this = (lhs->mul(rhs))->uminus();
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::mul_smul(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (p_impl) {
        if (lhs.p_impl) {
            p_impl->mul_smul(lhs, rhs);
        } else {
            p_impl->clear();
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBundleBase<BundleInterface, DerivedImpl>::algebra_t &AlgebraBundleBase<BundleInterface, DerivedImpl>::mul_sdiv(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (p_impl) {
        if (lhs.p_impl) {
            p_impl->mul_sdiv(lhs, rhs);
        } else {
            p_impl->clear();
        }
    }
    return downcast(*this);
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
std::ostream &AlgebraBundleBase<BundleInterface, DerivedImpl>::print(std::ostream &os) const {
    if (p_impl) {
        p_impl->print(os);
    } else {
        dtl::print_empty_algebra(os);
    }
    return os;
}
template <typename BundleInterface, template <typename, template <typename> class> class DerivedImpl>
bool AlgebraBundleBase<BundleInterface, DerivedImpl>::operator==(const algebra_t &other) const {
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
#endif// ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_
