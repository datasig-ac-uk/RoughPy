// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_IMPL_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_IMPL_H_

#include <roughpy/core/helpers.h>
#include <roughpy/core/traits.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/scalar_traits.h>

#include "algebra_base.h"
#include "algebra_info.h"
#include "algebra_iterator_impl.h"

#include <functional>

namespace rpy {
namespace algebra {

template <typename Interface, typename Impl>
class ImplAccessLayer : public Interface
{
protected:
    using Interface::Interface;

public:
    virtual Impl& get_data() noexcept = 0;
    virtual const Impl& get_data() const noexcept = 0;
    virtual Impl&& take_data() = 0;
};

template <typename Impl, typename Interface>
RPY_NO_UBSAN inline copy_cv_t<Impl, Interface>& algebra_cast(Interface& arg
) noexcept
{
    using access_t = copy_cv_t<ImplAccessLayer<Interface, Impl>, Interface>;
    static_assert(
            is_base_of<dtl::AlgebraInterfaceBase, Interface>::value,
            "casting to algebra implementation is only possible for "
            "interfaces"
    );
    //    RPY_DBG_ASSERT(dynamic_cast<access_t*>(&arg) != nullptr);
    return static_cast<access_t&>(arg).get_data();
}

template <typename Impl>
class OwnedStorageModel
{
    Impl m_data;

protected:
    static constexpr ImplementationType s_type = ImplementationType::Owned;

    template <typename... Args>
    explicit OwnedStorageModel(Args&&... args)
        : m_data(std::forward<Args>(args)...)
    {}
    explicit OwnedStorageModel(Impl&& arg) : m_data(std::move(arg)) {}
    explicit OwnedStorageModel(const Impl& arg) : m_data(arg) {}

    Impl& data() noexcept { return m_data; }
    const Impl& data() const noexcept { return m_data; }

    friend Impl&& take_data(OwnedStorageModel<Impl>&& storage)
    {
        return std::move(storage.m_data);
    }
};

template <typename Impl, typename Interface>
Impl&& take_algebra(typename Interface::algebra_t&& arg) noexcept;

template <typename Impl>
class BorrowedStorageModel
{
    Impl* p_data;

protected:
    static constexpr ImplementationType s_type = ImplementationType::Borrowed;

    explicit BorrowedStorageModel(Impl* arg) : p_data(arg) {}

    Impl& data()
    {
        if (is_const<Impl>::value) {
            RPY_THROW(
                    std::runtime_error,
                    "cannot get mutable data from const object"
            );
        }
        return *p_data;
    }
    const Impl& data() const noexcept { return *p_data; }
};

namespace dtl {

template <typename Impl>
class ConvertedArgument
{
    optional<Impl> m_holder;
    const Impl* p_ref = nullptr;

public:
    ConvertedArgument(Impl&& converted) : m_holder(std::move(converted))
    {
        p_ref = &*m_holder;
    }
    ConvertedArgument(const Impl& same_type) : m_holder(), p_ref(&same_type) {}
    operator const Impl&() const noexcept
    {
        RPY_DBG_ASSERT(p_ref != nullptr);
        return *p_ref;
    }
};

template <typename T>
using d_has_as_ptr_t = decltype(declval<const T&>().as_ptr());

#define RPY_HAS_FUSED_OP_CHECKER(NAME)                                         \
    template <typename T>                                                      \
    using d_##NAME = decltype(declval<T&>().NAME())

RPY_HAS_FUSED_OP_CHECKER(add_scal_prod);
RPY_HAS_FUSED_OP_CHECKER(sub_scal_prod);
RPY_HAS_FUSED_OP_CHECKER(add_scal_div);
RPY_HAS_FUSED_OP_CHECKER(sub_scal_div);
RPY_HAS_FUSED_OP_CHECKER(add_mul);
RPY_HAS_FUSED_OP_CHECKER(sub_mul);
RPY_HAS_FUSED_OP_CHECKER(mul_scal_prod);
RPY_HAS_FUSED_OP_CHECKER(mul_scal_div);

#undef RPY_HAS_FUSED_OP_CHECKER

struct no_implementation {
};
struct has_implementation {
};

template <template <typename> class MF, typename T>
using use_impl_t = conditional_t<
        is_detected<MF, T>::value, has_implementation, no_implementation>;

}// namespace dtl

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
class AlgebraImplementation : public ImplAccessLayer<Interface, Impl>,
                              protected StorageModel<Impl>
{
    using storage_base_t = StorageModel<Impl>;
    using access_layer_t = ImplAccessLayer<Interface, Impl>;

    static_assert(
            is_base_of<dtl::AlgebraInterfaceBase, Interface>::value,
            "algebra_interface must be an accessible base of Interface"
    );

    using alg_info = algebra_info<typename Interface::algebra_t, Impl>;
    using basis_traits = BasisInfo<
            typename Interface::basis_t, typename alg_info::basis_type>;

public:
    using interface_t = Interface;
    using algebra_t = typename Interface::algebra_t;
    using algebra_interface_t = typename Interface::algebra_interface_t;

    static constexpr AlgebraType s_alg_type = algebra_t::s_alg_type;

    using scalar_type = typename alg_info::scalar_type;
    using rational_type = typename alg_info::rational_type;
    using basis_type = typename alg_info::basis_type;

private:
    using iterator_type = AlgebraIterator<algebra_t>;
    using iterator_interface_type = AlgebraIteratorInterface<algebra_t>;
    using iterator_impl_type = AlgebraIteratorImplementation<
            algebra_t, basis_type, typename Impl::const_iterator>;

protected:
    using storage_base_t::data;

public:
    Impl& get_data() noexcept override { return data(); }
    const Impl& get_data() const noexcept override { return data(); }
    Impl&& take_data() override
    {
        if (is_same<storage_base_t, BorrowedStorageModel<Impl>>::value) {
            RPY_THROW(
                    std::runtime_error, "cannot take from a borrowed algebra"
            );
        }
        return std::move(data());
    }

    template <typename... Args>
    explicit AlgebraImplementation(context_pointer&& ctx, Args&&... args)
        : access_layer_t(
                std::move(ctx), alg_info::vtype(), alg_info::ctype(),
                storage_base_t::s_type
        ),
          storage_base_t(std::forward<Args>(args)...)
    {}

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

private:
    std::shared_ptr<iterator_interface_type>
    make_iterator_ptr(typename Impl::const_iterator it) const;

public:
    iterator_type begin() const override;
    iterator_type end() const override;

    void clear() override;
    void assign(const algebra_t& arg) override;

private:
    template <typename B>
    optional<scalars::ScalarArray>
    dense_data_impl(const B& data, integral_constant<bool, true>) const;

    template <typename B>
    optional<scalars::ScalarArray>
    dense_data_impl(const B& data, integral_constant<bool, false>) const;

public:
    optional<scalars::ScalarArray> dense_data() const override;

private:
    /*
     * The true implementation of operations needs a check of whether the
     * corresponding operation/function is defined for Impl. We're assuming
     * that, at a minimum, Impl has the standard arithmetic operations. But the
     * fused operations need not be defined.
     */
protected:
    dtl::ConvertedArgument<Impl> convert_argument(const algebra_t& arg) const;

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
    algebra_t uminus() const override;
    algebra_t add(const algebra_t& other) const override;
    algebra_t sub(const algebra_t& other) const override;
    algebra_t mul(const algebra_t& other) const override;
    algebra_t smul(const scalars::Scalar& scalar) const override;
    algebra_t sdiv(const scalars::Scalar& scalar) const override;

    void add_inplace(const algebra_t& other) override;
    void sub_inplace(const algebra_t& other) override;
    void mul_inplace(const algebra_t& other) override;
    void smul_inplace(const scalars::Scalar& other) override;
    void sdiv_inplace(const scalars::Scalar& other) override;

    void
    add_scal_mul(const algebra_t& arg, const scalars::Scalar& scalar) override;
    void
    sub_scal_mul(const algebra_t& arg, const scalars::Scalar& scalar) override;
    void
    add_scal_div(const algebra_t& arg, const scalars::Scalar& scalar) override;
    void
    sub_scal_div(const algebra_t& arg, const scalars::Scalar& scalar) override;

    void add_mul(const algebra_t& lhs, const algebra_t& rhs) override;
    void sub_mul(const algebra_t& lhs, const algebra_t& rhs) override;

    void mul_smul(const algebra_t& lhs, const scalars::Scalar& rhs) override;
    void mul_sdiv(const algebra_t& lhs, const scalars::Scalar& rhs) override;

    std::ostream& print(std::ostream& os) const override;
    bool equals(const algebra_t& other) const override;
};

template <typename Impl, typename Interface>
Impl&& take_algebra(typename Interface::algebra_t&& arg) noexcept
{
    return static_cast<ImplAccessLayer<Interface, Impl>&&>(*arg).take_data();
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
dimn_t AlgebraImplementation<Interface, Impl, StorageModel>::size() const
{
    return data().size();
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
dimn_t AlgebraImplementation<Interface, Impl, StorageModel>::dimension() const
{
    return data().dimension();
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
bool AlgebraImplementation<Interface, Impl, StorageModel>::is_zero() const
{
    // TODO: Replace with better, type aware implementation.
    return data().size() == 0;
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
optional<deg_t>
AlgebraImplementation<Interface, Impl, StorageModel>::degree() const
{
    return alg_info::degree(data());
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
optional<deg_t>
AlgebraImplementation<Interface, Impl, StorageModel>::width() const
{
    return basis_traits::width(&alg_info::basis(data()));
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
optional<deg_t>
AlgebraImplementation<Interface, Impl, StorageModel>::depth() const
{
    return basis_traits::depth(&alg_info::basis(data()));
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename Interface::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::clone() const
{
    return algebra_t(Interface::context(), data());
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename Interface::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::zero_like() const
{
    return algebra_t(Interface::context(), alg_info::create_like(data()));
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename AlgebraImplementation<Interface, Impl, StorageModel>::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::borrow() const
{
    //    return algebra_t(Interface::context(), &data());
    return algebra_t(Interface::context());
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename AlgebraImplementation<Interface, Impl, StorageModel>::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::borrow_mut()
{
    return algebra_t(Interface::context(), &data());
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
scalars::Scalar
AlgebraImplementation<Interface, Impl, StorageModel>::get(key_type key) const
{
    auto akey = basis_traits::convert_to_impl(&alg_info::basis(data()), key);
    using ref_t = decltype(data()[akey]);
    using trait = scalars::scalar_type_trait<ref_t>;
    return trait::make(data()[akey]);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
scalars::Scalar
AlgebraImplementation<Interface, Impl, StorageModel>::get_mut(key_type key)
{
    auto akey = basis_traits::convert_to_impl(&alg_info::basis(data()), key);
    using ref_t = decltype(data()[akey]);
    using trait = scalars::scalar_type_trait<ref_t>;
    return trait::make(data()[akey]);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
std::shared_ptr<AlgebraIteratorInterface<typename Interface::algebra_t>>
AlgebraImplementation<Interface, Impl, StorageModel>::make_iterator_ptr(
        typename Impl::const_iterator it
) const
{
    return std::shared_ptr<iterator_interface_type>(
            new iterator_impl_type(it, &alg_info::basis(data()))
    );
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
AlgebraIterator<typename Interface::algebra_t>
AlgebraImplementation<Interface, Impl, StorageModel>::begin() const
{
    return iterator_type(
            make_iterator_ptr(data().begin()), bit_cast<std::uintptr_t>(&data())
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
AlgebraIterator<typename Interface::algebra_t>
AlgebraImplementation<Interface, Impl, StorageModel>::end() const
{
    return iterator_type(
            make_iterator_ptr(data().end()), bit_cast<std::uintptr_t>(&data())
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::clear()
{
    data().clear();
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::assign(
        const algebra_t& arg
)
{
    data() = convert_argument(arg);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
optional<scalars::ScalarArray>
AlgebraImplementation<Interface, Impl, StorageModel>::dense_data() const
{
    using tag = integral_constant<
            bool,
            is_detected<dtl::d_has_as_ptr_t, decltype(data().base_vector())>::
                    value>;
    return dense_data_impl(data().base_vector(), tag());
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
template <typename B>
optional<scalars::ScalarArray>
AlgebraImplementation<Interface, Impl, StorageModel>::
        dense_data_impl(const B& data, integral_constant<bool, true>) const
{
    return scalars::ScalarArray{
            {Interface::coeff_type(), data.as_ptr()},
            data.dimension()
    };
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
template <typename B>
optional<scalars::ScalarArray>
AlgebraImplementation<Interface, Impl, StorageModel>::
        dense_data_impl(const B& data, integral_constant<bool, false>) const
{
    return Interface::dense_data();
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
dtl::ConvertedArgument<Impl>
AlgebraImplementation<Interface, Impl, StorageModel>::convert_argument(
        const algebra_t& arg
) const
{
    if (this->context() == arg->context()) {
        if (this->storage_type() == arg->storage_type()) {
            return algebra_cast<const Impl&>(*arg);
        }
        return take_algebra<Impl, Interface>(
                this->context()->convert(arg, this->storage_type())
        );
    }
    RPY_THROW(std::invalid_argument, "cannot convert argument");
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_mul_impl(
        const algebra_t& arg, const scalars::Scalar& scalar,
        dtl::no_implementation
)
{
    Interface::add_scal_mul(arg, scalar);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_mul_impl(
        const algebra_t& arg, const scalars::Scalar& scalar,
        dtl::no_implementation
)
{
    Interface::sub_scal_mul(arg, scalar);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_div_impl(
        const algebra_t& arg, const scalars::Scalar& scalar,
        dtl::no_implementation
)
{
    Interface::add_scal_div(arg, scalar);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_div_impl(
        const algebra_t& arg, const scalars::Scalar& scalar,
        dtl::no_implementation
)
{
    Interface::sub_scal_div(arg, scalar);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_mul_impl(
        const algebra_t& lhs, const algebra_t& rhs, dtl::no_implementation
)
{
    Interface::add_mul(lhs, rhs);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_mul_impl(
        const algebra_t& lhs, const algebra_t& rhs, dtl::no_implementation
)
{
    Interface::sub_mul(lhs, rhs);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_smul_impl(
        const algebra_t& lhs, const scalars::Scalar& rhs, dtl::no_implementation
)
{
    Interface::mul_smul(lhs, rhs);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_sdiv_impl(
        const algebra_t& lhs, const scalars::Scalar& rhs, dtl::no_implementation
)
{
    Interface::mul_sdiv(lhs, rhs);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_mul_impl(
        const algebra_t& arg, const scalars::Scalar& scalar,
        dtl::has_implementation
)
{
    data().add_scal_prod(
            convert_argument(arg), scalars::scalar_cast<scalar_type>(scalar)
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_mul_impl(
        const algebra_t& arg, const scalars::Scalar& scalar,
        dtl::has_implementation
)
{
    data().sub_scal_prod(
            convert_argument(arg), scalars::scalar_cast<scalar_type>(scalar)
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_div_impl(
        const algebra_t& arg, const scalars::Scalar& scalar,
        dtl::has_implementation
)
{
    data().add_scal_div(
            convert_argument(arg), scalars::scalar_cast<rational_type>(scalar)
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_div_impl(
        const algebra_t& arg, const scalars::Scalar& scalar,
        dtl::has_implementation
)
{
    data().sub_scal_sub(
            convert_argument(arg), scalars::scalar_cast<rational_type>(scalar)
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_mul_impl(
        const algebra_t& lhs, const algebra_t& rhs, dtl::has_implementation
)
{
    data().add_mul(convert_argument(lhs), convert_argument(rhs));
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_mul_impl(
        const algebra_t& lhs, const algebra_t& rhs, dtl::has_implementation
)
{
    data().sub_mul(convert_argument(lhs), convert_argument(rhs));
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_smul_impl(
        const algebra_t& lhs, const scalars::Scalar& rhs,
        dtl::has_implementation
)
{
    data().mul_scal_prod(
            convert_argument(lhs), scalars::scalar_cast<scalar_type>(rhs)
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_sdiv_impl(
        const algebra_t& lhs, const scalars::Scalar& rhs,
        dtl::has_implementation
)
{
    data().mul_scal_div(
            convert_argument(lhs), scalars::scalar_cast<rational_type>(rhs)
    );
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename Interface::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::uminus() const
{
    return algebra_t(Interface::context(), -data());
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename Interface::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::add(const algebra_t& other
) const
{
    std::plus<Impl> plus;
    return algebra_t(
            Interface::context(), plus(data(), convert_argument(other))
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename Interface::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::sub(const algebra_t& other
) const
{
    std::minus<Impl> minus;
    return algebra_t(
            Interface::context(), minus(data(), convert_argument(other))
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename Interface::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::mul(const algebra_t& other
) const
{
    std::multiplies<Impl> mul;
    return algebra_t(
            Interface::context(),
            mul(data(), static_cast<const Impl&>(convert_argument(other)))
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename Interface::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::smul(
        const scalars::Scalar& scalar
) const
{
    return algebra_t(
            Interface::context(),
            data() * scalars::scalar_cast<scalar_type>(scalar)
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
typename Interface::algebra_t
AlgebraImplementation<Interface, Impl, StorageModel>::sdiv(
        const scalars::Scalar& scalar
) const
{
    return algebra_t(
            Interface::context(),
            data() / scalars::scalar_cast<rational_type>(scalar)
    );
}

namespace ADL_FORCE {

template <typename T, typename S>
void add_assign(T& left, const S& right)
{
    left += static_cast<const T&>(right);
}

template <typename T, typename S>
void sub_assign(T& left, const S& right)
{
    left -= static_cast<const T&>(right);
}

}// namespace ADL_FORCE

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_inplace(
        const algebra_t& other
)
{
    ADL_FORCE::add_assign(data(), convert_argument(other));
    //    data() += convert_argument(other);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_inplace(
        const algebra_t& other
)
{
    ADL_FORCE::sub_assign(data(), convert_argument(other));
    //    data() -= convert_argument(other);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_inplace(
        const algebra_t& other
)
{
    data() *= static_cast<const Impl&>(convert_argument(other));
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::smul_inplace(
        const scalars::Scalar& other
)
{
    data() *= scalars::scalar_cast<scalar_type>(other);
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sdiv_inplace(
        const scalars::Scalar& other
)
{
    data() /= scalars::scalar_cast<rational_type>(other);
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_mul(
        const algebra_t& arg, const scalars::Scalar& scalar
)
{
    add_scal_mul_impl(
            arg, scalar, dtl::use_impl_t<dtl::d_add_scal_prod, Impl>()
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_mul(
        const algebra_t& arg, const scalars::Scalar& scalar
)
{
    sub_scal_mul_impl(
            arg, scalar, dtl::use_impl_t<dtl::d_sub_scal_prod, Impl>()
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_div(
        const algebra_t& arg, const scalars::Scalar& scalar
)
{
    add_scal_div_impl(
            arg, scalar, dtl::use_impl_t<dtl::d_add_scal_div, Impl>()
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_div(
        const algebra_t& arg, const scalars::Scalar& scalar
)
{
    sub_scal_div_impl(
            arg, scalar, dtl::use_impl_t<dtl::d_sub_scal_div, Impl>()
    );
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    add_mul_impl(lhs, rhs, dtl::use_impl_t<dtl::d_add_mul, Impl>());
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_mul(
        const algebra_t& lhs, const algebra_t& rhs
)
{
    sub_mul_impl(lhs, rhs, dtl::use_impl_t<dtl::d_sub_mul, Impl>());
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_smul(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    mul_smul_impl(lhs, rhs, dtl::use_impl_t<dtl::d_mul_scal_prod, Impl>());
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_sdiv(
        const algebra_t& lhs, const scalars::Scalar& rhs
)
{
    mul_sdiv_impl(lhs, rhs, dtl::use_impl_t<dtl::d_mul_scal_div, Impl>());
}

template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
std::ostream&
AlgebraImplementation<Interface, Impl, StorageModel>::print(std::ostream& os
) const
{
    return os << data();
}
template <
        typename Interface, typename Impl,
        template <typename> class StorageModel>
bool AlgebraImplementation<Interface, Impl, StorageModel>::equals(
        const algebra_t& other
) const
{
    return data() == static_cast<const Impl&>(convert_argument(other));
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_IMPL_H_
