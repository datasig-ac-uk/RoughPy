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

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_IMPL_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_IMPL_H_

#include "algebra_bundle.h"
#include "algebra_impl.h"

#include <roughpy/core/traits.h>

namespace rpy {
namespace algebra {

template <typename Bundle>
struct bundle_traits;

template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
class AlgebraBundleImplementation
    : protected StorageModel<BundleImpl>,
      public ImplAccessLayer<Interface, BundleImpl>
{

    using storage_base_t = StorageModel<BundleImpl>;
    using access_layer_t = ImplAccessLayer<Interface, BundleImpl>;

    using algebra_t = typename Interface::algebra_t;
    using base_alg_t = typename Interface::base_t;
    using fibre_alg_t = typename Interface::fibre_t;

    using base_interface_t = typename Interface::base_interface_t;
    using fibre_interface_t = typename Interface::fibre_interface_t;

    using bundle_traits_t = bundle_traits<BundleImpl>;

    using real_base_t = typename bundle_traits_t::base_type;
    using real_fibre_t = typename bundle_traits_t::fibre_type;

    using base_impl_t = AlgebraImplementation<
            base_interface_t, real_base_t, BorrowedStorageModel>;
    using fibre_impl_t = AlgebraImplementation<
            fibre_interface_t, real_fibre_t, BorrowedStorageModel>;

    using scalar_type = typename bundle_traits_t::scalar_type;
    using rational_type = typename bundle_traits_t::rational_type;

protected:
    using storage_base_t::data;

public:
    BundleImpl& get_data() noexcept override { return data(); }

    const BundleImpl& get_data() const noexcept override { return data(); }

    BundleImpl&& take_data() override
    {
        if RPY_IF_CONSTEXPR (is_same<storage_base_t,
                                     BorrowedStorageModel<BundleImpl>>::value) {
            RPY_THROW(std::runtime_error, "cannot take from a borrowed algebra");
        } else {
            return std::move(data());
        }
    }

    template <typename... Args>
    explicit AlgebraBundleImplementation(context_pointer&& ctx, Args&&... args)
        : storage_base_t(std::forward<Args>(args)...),
          access_layer_t(std::move(ctx))
    {}

protected:
    dtl::ConvertedArgument<BundleImpl> convert_argument(const algebra_t& arg
    ) const;

public:
    dimn_t size() const override;
    dimn_t dimension() const override;
    bool is_zero() const override;
    optional<deg_t> degree() const override;
    optional<deg_t> width() const override;
    optional<deg_t> depth() const override;

    algebra_t clone() const override;
    algebra_t zero_like() const override;

    algebra_t borrow() const override;
    algebra_t borrow_mut() override;

    scalars::Scalar get(key_type key) const override;
    scalars::Scalar get_mut(key_type key) override;

    base_alg_t base() override;
    fibre_alg_t fibre() override;

    std::ostream& print(std::ostream& os) const override;

    bool equals(const algebra_t& other) const override;

    algebra_t uminus() const override;
    algebra_t add(const algebra_t& other) const override;
    algebra_t sub(const algebra_t& other) const override;
    algebra_t mul(const algebra_t& other) const override;
    algebra_t smul(const scalars::Scalar& other) const override;
    algebra_t sdiv(const scalars::Scalar& other) const override;

    void add_inplace(const algebra_t& other) override;
    void sub_inplace(const algebra_t& other) override;
    void mul_inplace(const algebra_t& other) override;
    void smul_inplace(const scalars::Scalar& other) override;
    void sdiv_inplace(const scalars::Scalar& other) override;

private:
    void add_scal_mul_impl(
            const algebra_t& arg, const scalars::Scalar& scalar,
            dtl::no_implementation
    );
    void sub_scal_mul_impl(
            const algebra_t& arg, const scalars::Scalar& scalar,
            dtl::no_implementation
    );
    void add_scal_div_impl(
            const algebra_t& arg, const scalars::Scalar& scalar,
            dtl::no_implementation
    );
    void sub_scal_div_impl(
            const algebra_t& arg, const scalars::Scalar& scalar,
            dtl::no_implementation
    );

    void add_mul_impl(
            const algebra_t& lhs, const algebra_t& rhs, dtl::no_implementation
    );
    void sub_mul_impl(
            const algebra_t& lhs, const algebra_t& rhs, dtl::no_implementation
    );

    void mul_smul_impl(
            const algebra_t& lhs, const scalars::Scalar& rhs,
            dtl::no_implementation
    );
    void mul_sdiv_impl(
            const algebra_t& lhs, const scalars::Scalar& rhs,
            dtl::no_implementation
    );

    void add_scal_mul_impl(
            const algebra_t& arg, const scalars::Scalar& scalar,
            dtl::has_implementation
    );
    void sub_scal_mul_impl(
            const algebra_t& arg, const scalars::Scalar& scalar,
            dtl::has_implementation
    );
    void add_scal_div_impl(
            const algebra_t& arg, const scalars::Scalar& scalar,
            dtl::has_implementation
    );
    void sub_scal_div_impl(
            const algebra_t& arg, const scalars::Scalar& scalar,
            dtl::has_implementation
    );

    void add_mul_impl(
            const algebra_t& lhs, const algebra_t& rhs, dtl::has_implementation
    );
    void sub_mul_impl(
            const algebra_t& lhs, const algebra_t& rhs, dtl::has_implementation
    );

    void mul_smul_impl(
            const algebra_t& lhs, const scalars::Scalar& rhs,
            dtl::has_implementation
    );
    void mul_sdiv_impl(
            const algebra_t& lhs, const scalars::Scalar& rhs,
            dtl::has_implementation
    );

public:
    void
    add_scal_mul(const algebra_t& rhs, const scalars::Scalar& scalar) override;
    void
    sub_scal_mul(const algebra_t& rhs, const scalars::Scalar& scalar) override;
    void
    add_scal_div(const algebra_t& rhs, const scalars::Scalar& scalar) override;
    void
    sub_scal_div(const algebra_t& rhs, const scalars::Scalar& scalar) override;

    void add_mul(const algebra_t& lhs, const algebra_t& rhs) override;
    void sub_mul(const algebra_t& lhs, const algebra_t& rhs) override;
    void mul_smul(const algebra_t& lhs, const scalars::Scalar& rhs) override;
    void mul_sdiv(const algebra_t& lhs, const scalars::Scalar& rhs) override;
};

template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
dtl::ConvertedArgument<BundleImpl>
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        convert_argument(const algebra_t& arg) const
{
    RPY_CHECK(this->context() == arg->context());
    if (this->storage_type() == arg->storage_type()) {
        return algebra_cast<const BundleImpl&>(*arg);
    }
    return take_algebra<BundleImpl, Interface>(
            this->context()->convert(arg, this->storage_type())
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
dimn_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::size() const
{
    // TODO: use traits
    return data().size();
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
dimn_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::dimension(
) const
{
    // TODO: use traits
    return data.dimension();
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
bool AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::is_zero(
) const
{
    return data().dimension() == 0
            || data() == bundle_traits_t::zero_like(data());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
optional<deg_t>
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::degree() const
{
    return bundle_traits_t::degree(data());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
optional<deg_t>
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::width() const
{
    // TODO: IMplementation needed
    return {};
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
optional<deg_t>
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::depth() const
{
    // TODO: Implementation needed
    return {};
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::clone() const
{
    return algebra_t(Interface::context(), data());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::zero_like(
) const
{
    return bundle_traits_t::zero_like(data());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::borrow() const
{
    return algebra_t(Interface::context());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::borrow_mut()
{
    return algebra_t(Interface::context(), &data());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
scalars::Scalar
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::get(
        key_type key
) const
{
    // TODO: needs implementation
    return scalars::Scalar();
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
scalars::Scalar
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::get_mut(
        key_type key
)
{
    // TODO: Needs implementation
    return scalars::Scalar();
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::base_alg_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::base()
{
    return base_alg_t(Interface::context(), &data().base());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::fibre_alg_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::fibre()
{
    return fibre_alg_t(Interface::context(), &data().fibre());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
std::ostream&
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::print(
        std::ostream& os
) const
{
    return os << data();
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
bool AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::equals(
        const algebra_t& other
) const
{
    return data() == static_cast<const BundleImpl&>(convert_argument(other));
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::uminus() const
{
    return algebra_t(Interface::context(), -data());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::add(
        const algebra_t& other
) const
{
    std::plus<BundleImpl> plus;
    return algebra_t(
            Interface::context(), plus(data(), convert_argument(other))
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::sub(
        const algebra_t& other
) const
{
    std::minus<BundleImpl> minus;
    return algebra_t(
            Interface::context(), minus(data(), convert_argument(other))
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::mul(
        const algebra_t& other
) const
{
    std::multiplies<BundleImpl> mul;
    return algebra_t(
            Interface::context(), mul(data(), convert_argument(other))
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::smul(
        const scalars::Scalar& other
) const
{
    return algebra_t(
            Interface::context(),
            data() * scalars::scalar_cast<scalar_type>(other)
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
typename AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::algebra_t
AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::sdiv(
        const scalars::Scalar& other
) const
{
    return algebra_t(
            Interface::context(),
            data() / scalars::scalar_cast<rational_type>(other)
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::add_inplace(const algebra_t& other
)
{
    ADL_FORCE::add_assign(data(), convert_argument(other));
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::sub_inplace(const algebra_t& other
)
{
    ADL_FORCE::sub_assign(data(), convert_argument(other));
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<
        Interface, BundleImpl, StorageModel>::mul_inplace(const algebra_t& other
)
{
    data() *= static_cast<const BundleImpl&>(convert_argument(other));
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        smul_inplace(const scalars::Scalar& other)
{
    data() *= scalars::scalar_cast<scalar_type>(other);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sdiv_inplace(const scalars::Scalar& other)
{
    data() /= scalars::scalar_cast<rational_type>(other);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        add_scal_mul_impl(
                const algebra_t& arg, const scalars::Scalar& scalar,
                dtl::no_implementation
        )
{
    Interface::add_scal_mul(arg, scalar);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sub_scal_mul_impl(
                const algebra_t& arg, const scalars::Scalar& scalar,
                dtl::no_implementation
        )
{
    Interface::sub_scal_mul(arg, scalar);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        add_scal_div_impl(
                const algebra_t& arg, const scalars::Scalar& scalar,
                dtl::no_implementation
        )
{
    Interface::add_scal_div(arg, scalar);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sub_scal_div_impl(
                const algebra_t& arg, const scalars::Scalar& scalar,
                dtl::no_implementation
        )
{
    Interface::sub_scal_div(arg, scalar);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        add_mul_impl(
                const algebra_t& lhs, const algebra_t& rhs,
                dtl::no_implementation
        )
{
    Interface::add_mul(lhs, rhs);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sub_mul_impl(
                const algebra_t& lhs, const algebra_t& rhs,
                dtl::no_implementation
        )
{
    Interface::sub_mul(lhs, rhs);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        mul_smul_impl(
                const algebra_t& lhs, const scalars::Scalar& rhs,
                dtl::no_implementation
        )
{
    Interface::mul_smul(lhs, rhs);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        mul_sdiv_impl(
                const algebra_t& lhs, const scalars::Scalar& rhs,
                dtl::no_implementation
        )
{
    Interface::mul_sdiv(lhs, rhs);
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        add_scal_mul_impl(
                const algebra_t& arg, const scalars::Scalar& scalar,
                dtl::has_implementation
        )
{
    data().add_scal_prod(
            convert_argument(arg), scalars::scalar_cast<scalar_type>(scalar)
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sub_scal_mul_impl(
                const algebra_t& arg, const scalars::Scalar& scalar,
                dtl::has_implementation
        )
{
    data().sub_scal_prod(
            convert_argument(arg), scalars::scalar_cast<scalar_type>(scalar)
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        add_scal_div_impl(
                const algebra_t& arg, const scalars::Scalar& scalar,
                dtl::has_implementation
        )
{
    data().add_scal_div(
            convert_argument(arg), scalars::scalar_cast<rational_type>(scalar)
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sub_scal_div_impl(
                const algebra_t& arg, const scalars::Scalar& scalar,
                dtl::has_implementation
        )
{
    data().sub_scal_sub(
            convert_argument(arg), scalars::scalar_cast<rational_type>(scalar)
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        add_mul_impl(
                const algebra_t& lhs, const algebra_t& rhs,
                dtl::has_implementation
        )
{}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sub_mul_impl(
                const algebra_t& lhs, const algebra_t& rhs,
                dtl::has_implementation
        )
{
    data().add_mul(convert_argument(lhs), convert_argument(rhs));
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        mul_smul_impl(
                const algebra_t& lhs, const scalars::Scalar& rhs,
                dtl::has_implementation
        )
{
    data().sub_mul(convert_argument(lhs), convert_argument(rhs));
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        mul_sdiv_impl(
                const algebra_t& lhs, const scalars::Scalar& rhs,
                dtl::has_implementation
        )
{
    data().mul_scal_prod(
            convert_argument(lhs), scalars::scalar_cast<scalar_type>(rhs)
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        add_scal_mul(const algebra_t& rhs, const scalars::Scalar& scalar)
{
    add_scal_mul_impl(
            rhs, scalar, dtl::use_impl_t<dtl::d_add_scal_prod, BundleImpl>()
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sub_scal_mul(const algebra_t& rhs, const scalars::Scalar& scalar)
{
    sub_scal_mul_impl(
            rhs, scalar, dtl::use_impl_t<dtl::d_sub_scal_prod, BundleImpl>()
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        add_scal_div(const algebra_t& rhs, const scalars::Scalar& scalar)
{
    add_scal_div_impl(
            rhs, scalar, dtl::use_impl_t<dtl::d_add_scal_div, BundleImpl>()
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::
        sub_scal_div(const algebra_t& rhs, const scalars::Scalar& scalar)
{
    sub_scal_div_impl(
            rhs, scalar, dtl::use_impl_t<dtl::d_sub_scal_div, BundleImpl>()
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::add_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    add_mul_impl(lhs, rhs, dtl::use_impl_t<dtl::d_add_mul, BundleImpl>());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::sub_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    sub_mul_impl(lhs, rhs, dtl::use_impl_t<dtl::d_sub_mul, BundleImpl>());
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::mul_smul(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    mul_smul_impl(
            lhs, rhs, dtl::use_impl_t<dtl::d_mul_scal_prod, BundleImpl>()
    );
}
template <
        typename Interface, typename BundleImpl,
        template <typename> class StorageModel>
void AlgebraBundleImplementation<Interface, BundleImpl, StorageModel>::mul_sdiv(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    mul_sdiv_impl(lhs, rhs, dtl::use_impl_t<dtl::d_mul_scal_div, BundleImpl>());
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_IMPL_H_
