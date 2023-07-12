// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_

#include "algebra_base.h"
#include "algebra_fwd.h"

#include <roughpy/platform/serialization.h>

RPY_WARNING_PUSH
RPY_MSVC_DISABLE_WARNING(4661)

namespace rpy {
namespace algebra {

template <typename Bundle, typename Base, typename Fibre>
class  BundleInterface
    : public dtl::algebra_base_resolution<
              Bundle, typename Base::basis_type, dtl::AlgebraArithmetic,
              dtl::AlgebraElementAccess>::type
{
public:
    using base_alg_t = Base;
    using fibre_alg_t = Fibre;
    using base_interface_t = typename Base::interface_t;
    using fibre_interface_t = typename Fibre::interface_t;

    virtual base_alg_t base() = 0;
    virtual fibre_alg_t fibre() = 0;
};

template <
        typename BundleInterface,
        template <typename, template <typename> class> class DerivedImpl
        = dtl::with_interface<BundleInterface>::template type>
class AlgebraBundleBase
{

    explicit AlgebraBundleBase(std::unique_ptr<BundleInterface>&& impl)
        : p_impl(std::move(impl))
    {}

    explicit AlgebraBundleBase(BundleInterface* impl) : p_impl(impl) {}

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

    AlgebraBundleBase(const AlgebraBundleBase& other);
    AlgebraBundleBase(AlgebraBundleBase&& other) noexcept = default;

    AlgebraBundleBase& operator=(const AlgebraBundleBase& other);
    AlgebraBundleBase& operator=(AlgebraBundleBase&& other) noexcept = default;

    RPY_NO_DISCARD
    algebra_t borrow() const;
    RPY_NO_DISCARD
    algebra_t borrow_mut();

    RPY_NO_DISCARD
    const BundleInterface& operator*() const noexcept { return *p_impl; }
    RPY_NO_DISCARD
    BundleInterface& operator*() noexcept { return *p_impl; }
    RPY_NO_DISCARD
    const BundleInterface* operator->() const noexcept { return p_impl.get(); }
    RPY_NO_DISCARD
    BundleInterface* operator->() noexcept { return p_impl.get(); }

    RPY_NO_DISCARD
    constexpr operator bool() const noexcept
    {
        return static_cast<bool>(p_impl);
    }

    RPY_NO_DISCARD
    dimn_t dimension() const;
    RPY_NO_DISCARD
    dimn_t size() const;
    RPY_NO_DISCARD
    bool is_zero() const;
    RPY_NO_DISCARD
    optional<deg_t> width() const;
    RPY_NO_DISCARD
    optional<deg_t> depth() const;
    RPY_NO_DISCARD
    optional<deg_t> degree() const;

    RPY_NO_DISCARD
    VectorType storage_type() const noexcept;
    RPY_NO_DISCARD
    const scalars::ScalarType* coeff_type() const noexcept;

    RPY_NO_DISCARD
    scalars::Scalar operator[](key_type k) const;
    RPY_NO_DISCARD
    scalars::Scalar operator[](key_type k);

protected:
    RPY_NO_DISCARD
    static algebra_t& downcast(AlgebraBundleBase& arg)
    {
        return static_cast<algebra_t&>(arg);
    }
    RPY_NO_DISCARD
    static const algebra_t& downcast(const AlgebraBundleBase& arg)
    {
        return static_cast<const algebra_t&>(arg);
    }

public:
    RPY_NO_DISCARD
    algebra_t uminus() const;
    RPY_NO_DISCARD
    algebra_t add(const algebra_t& rhs) const;
    RPY_NO_DISCARD
    algebra_t sub(const algebra_t& rhs) const;
    RPY_NO_DISCARD
    algebra_t mul(const algebra_t& rhs) const;
    RPY_NO_DISCARD
    algebra_t smul(const scalars::Scalar& rhs) const;
    RPY_NO_DISCARD
    algebra_t sdiv(const scalars::Scalar& rhs) const;

    algebra_t& add_inplace(const algebra_t& rhs);
    algebra_t& sub_inplace(const algebra_t& rhs);
    algebra_t& mul_inplace(const algebra_t& rhs);
    algebra_t& smul_inplace(const scalars::Scalar& rhs);
    algebra_t& sdiv_inplace(const scalars::Scalar& rhs);

    algebra_t& add_scal_mul(const algebra_t& lhs, const scalars::Scalar& rhs);
    algebra_t& sub_scal_mul(const algebra_t& lhs, const scalars::Scalar& rhs);
    algebra_t& add_scal_div(const algebra_t& lhs, const scalars::Scalar& rhs);
    algebra_t& sub_scal_div(const algebra_t& lhs, const scalars::Scalar& rhs);

    algebra_t& add_mul(const algebra_t& lhs, const algebra_t& rhs);
    algebra_t& sub_mul(const algebra_t& lhs, const algebra_t& rhs);
    algebra_t& mul_smul(const algebra_t& lhs, const scalars::Scalar& rhs);
    algebra_t& mul_sdiv(const algebra_t& lhs, const scalars::Scalar& rhs);

    std::ostream& print(std::ostream& os) const;

    RPY_NO_DISCARD
    bool operator==(const algebra_t& other) const;
    RPY_NO_DISCARD
    bool operator!=(const algebra_t& other) const { return !operator==(other); }

    // #ifndef RPY_DISABLE_SERIALIZATION
    // private:
    //     friend rpy::serialization_access;
    //
    //     RPY_SERIAL_SPLIT_MEMBER();
    //
    //     template <typename Ar>
    //     void save(Ar &ar, const unsigned int /*version*/) const {
    //         context_pointer ctx = (p_impl) ? p_impl->context() : nullptr;
    //         auto spec = get_context_spec(ctx);
    //         ar << spec.width;
    //         ar << spec.depth;
    //         ar << spec.stype_id;
    //         ar << spec.backend;
    //         ar << algebra_t::s_alg_type;
    //         ar << alg_to_raw_bytes(ctx, algebra_t::s_alg_type, p_impl.get());
    //     }
    //
    //     template <typename Ar>
    //     void load(Ar &ar, const unsigned int /*version*/) {
    //         BasicContextSpec spec;
    //         ar >> spec.width;
    //         ar >> spec.depth;
    //         ar >> spec.stype_id;
    //         ar >> spec.backend;
    //
    //         auto ctx = from_context_spec(spec);
    //
    //         AlgebraType atype;
    //         ar >> atype;
    //         std::vector<byte> raw_data;
    //         ar >> raw_data;
    //         UnspecifiedAlgebraType alg = alg_from_raw_bytes(ctx, atype,
    //         raw_data);
    //
    //         RPY_CHECK(algebra_t::s_alg_type == atype);
    //         p_impl =
    //         std::unique_ptr<BundleInterface>(reinterpret_cast<BundleInterface
    //         *>(alg /*.release()*/));
    //     }
    //
    // #endif

    RPY_SERIAL_LOAD_FN();
    RPY_SERIAL_SAVE_FN();
};

}// namespace algebra
}// namespace rpy

RPY_WARNING_POP

#endif// ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_
