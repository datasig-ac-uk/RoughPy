// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_

#include "algebra_fwd.h"
#include "context_fwd.h"

#include <memory>

#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>

#include "algebra_iterator.h"
#include "basis.h"
#include "interface_traits.h"


RPY_WARNING_PUSH
RPY_MSVC_DISABLE_WARNING(4661)

namespace rpy {
namespace algebra {


// Forward declarations of implementation templates
template <typename>
class OwnedStorageModel;

template <typename>
class BorrowedStorageModel;

template <typename, typename, template <typename> class>
class AlgebraImplementation;

namespace dtl {
template <
        typename Impl,
        template <typename, template <typename> class> class Wrapper>
using select_owned_or_borrowed_t = conditional_t<
        is_pointer<remove_reference_t<Impl>>::value,
        Wrapper<remove_cv_t<remove_pointer_t<Impl>>, BorrowedStorageModel>,
        Wrapper<remove_cv_ref_t<Impl>, OwnedStorageModel>>;

template <typename IFace>
struct with_interface {
    template <typename Impl, template <typename> class StorageModel>
    using type = AlgebraImplementation<IFace, Impl, StorageModel>;
};

ROUGHPY_ALGEBRA_EXPORT void print_empty_algebra(std::ostream& os);
ROUGHPY_ALGEBRA_EXPORT const scalars::ScalarType*
context_to_scalars(const context_pointer& ptr);

}// namespace dtl

/**
 * @brief Base wrapper for algebra types
 * @tparam Interface Interface of algebra type
 * @tparam DerivedImpl Optional specialised template wrapper, use if
 *  AlgebraImplementation is not sufficient for the algebra interface.
 */
template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl
        = dtl::with_interface<Interface>::template type>
class AlgebraBase
{

protected:
    std::unique_ptr<Interface> p_impl;
    friend struct algebra_access<Interface>;


public:
    explicit AlgebraBase(std::unique_ptr<Interface> impl)
        : p_impl(std::move(impl))
    {}
    explicit AlgebraBase(UnspecifiedAlgebraType&& impl)
            : p_impl(reinterpret_cast<Interface*>(impl.release()))
    {}

    explicit AlgebraBase(Interface* impl) : p_impl(impl) {}

    ~AlgebraBase();

    using interface_t = Interface;

    using basis_type = traits::basis_of<Interface>;
    using key_type = traits::key_of<Interface>;

    using algebra_t = traits::algebra_of<Interface>;
    using const_iterator = AlgebraIterator<algebra_t>;

    AlgebraBase() : p_impl(nullptr) {}
    AlgebraBase(const AlgebraBase& other);
    AlgebraBase(AlgebraBase&& other) noexcept;

    AlgebraBase& operator=(const AlgebraBase& other);
    AlgebraBase& operator=(AlgebraBase&& other) noexcept;

    explicit AlgebraBase(context_pointer ctx);

    template <
            typename Impl,
            typename
            = enable_if_t<!is_same<remove_cv_ref_t<Impl>, algebra_t>::value>>
    explicit AlgebraBase(context_pointer ctx, Impl&& arg)
        : p_impl(new dtl::select_owned_or_borrowed_t<Impl, DerivedImpl>(
                std::move(ctx), std::forward<Impl>(arg)
        ))
    {}

    template <typename Impl, typename... Args>
    static enable_if_t<!is_base_of<Interface, Impl>::value, algebra_t>
    from_args(context_pointer ctx, Args&&... args)
    {
        return algebra_t(
                std::move(ctx),
                new dtl::select_owned_or_borrowed_t<Impl, DerivedImpl>(
                        std::forward<Args>(args)...
                )
        );
    }

    template <typename Wrapper, typename... Args>
    static enable_if_t<is_base_of<Interface, Wrapper>::value, algebra_t>
    from_args(context_pointer ctx, Args&&... args)
    {
        return algebra_t(
                std::move(ctx), new Wrapper(std::forward<Args>(args)...)
        );
    }

    RPY_NO_DISCARD
    algebra_t borrow() const;
    RPY_NO_DISCARD
    algebra_t borrow_mut();

    RPY_NO_DISCARD
    const Interface& operator*() const noexcept { return *p_impl; }
    RPY_NO_DISCARD
    Interface& operator*() noexcept { return *p_impl; }
    RPY_NO_DISCARD
    const Interface* operator->() const noexcept { return p_impl.get(); }
    RPY_NO_DISCARD
    Interface* operator->() noexcept { return p_impl.get(); }

    RPY_NO_DISCARD
    constexpr operator bool() const noexcept
    {
        return static_cast<bool>(p_impl);
    }

    RPY_NO_DISCARD
    explicit inline operator const Interface*() const noexcept
    {
        return p_impl.get();
    }

    RPY_NO_DISCARD
    explicit inline operator Interface*() noexcept
    {
        return p_impl.get();
    }

    RPY_NO_DISCARD
    basis_type basis() const;
    RPY_NO_DISCARD
    dimn_t dimension() const;
    RPY_NO_DISCARD
    dimn_t size() const;
    RPY_NO_DISCARD
    bool is_zero() const;
    RPY_NO_DISCARD
    context_pointer context() const noexcept;
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

    RPY_NO_DISCARD
    const_iterator begin() const;
    RPY_NO_DISCARD
    const_iterator end() const;

    RPY_NO_DISCARD
    optional<scalars::ScalarArray> dense_data() const;

protected:
    RPY_NO_DISCARD
    static bool is_equivalent_to_zero(const AlgebraBase& arg)
    {
        // For the moment, we just check if the arg has a null-p_impl
        // In the future we might do something else.
        return arg.p_impl == nullptr;
    }

    RPY_NO_DISCARD
    static algebra_t& downcast(AlgebraBase& arg)
    {
        return static_cast<algebra_t&>(arg);
    }
    RPY_NO_DISCARD
    static const algebra_t& downcast(const AlgebraBase& arg)
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

    RPY_NO_DISCARD
    algebra_t& add_inplace(const algebra_t& rhs);
    RPY_NO_DISCARD
    algebra_t& sub_inplace(const algebra_t& rhs);
    RPY_NO_DISCARD
    algebra_t& mul_inplace(const algebra_t& rhs);
    RPY_NO_DISCARD
    algebra_t& smul_inplace(const scalars::Scalar& rhs);
    RPY_NO_DISCARD
    algebra_t& sdiv_inplace(const scalars::Scalar& rhs);

    RPY_NO_DISCARD
    algebra_t& add_scal_mul(const algebra_t& lhs, const scalars::Scalar& rhs);
    RPY_NO_DISCARD
    algebra_t& sub_scal_mul(const algebra_t& lhs, const scalars::Scalar& rhs);
    RPY_NO_DISCARD
    algebra_t& add_scal_div(const algebra_t& lhs, const scalars::Scalar& rhs);
    RPY_NO_DISCARD
    algebra_t& sub_scal_div(const algebra_t& lhs, const scalars::Scalar& rhs);

    RPY_NO_DISCARD
    algebra_t& add_mul(const algebra_t& lhs, const algebra_t& rhs);
    RPY_NO_DISCARD
    algebra_t& sub_mul(const algebra_t& lhs, const algebra_t& rhs);
    RPY_NO_DISCARD
    algebra_t& mul_smul(const algebra_t& lhs, const scalars::Scalar& rhs);
    RPY_NO_DISCARD
    algebra_t& mul_sdiv(const algebra_t& lhs, const scalars::Scalar& rhs);

    std::ostream& print(std::ostream& os) const;

    RPY_NO_DISCARD
    bool operator==(const algebra_t& other) const;
    RPY_NO_DISCARD
    bool operator!=(const algebra_t& other) const { return !operator==(other); }

private:
    RPY_SERIAL_ACCESS();

    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();
};


template <
        typename Interface,
        template <typename, template <typename> class> class DerivedImpl>
inline std::ostream&
operator<<(std::ostream& os, const AlgebraBase<Interface, DerivedImpl>& alg)
{
    return alg.print(os);
}

namespace dtl {

ROUGHPY_ALGEBRA_EXPORT
UnspecifiedAlgebraType
try_create_new_empty(context_pointer ctx, AlgebraType alg_type);

template <typename Interface>
std::unique_ptr<Interface> downcast_interface_ptr(UnspecifiedAlgebraType ptr)
{
    return std::unique_ptr<Interface>(reinterpret_cast<Interface*>(ptr.release()
    ));
}

ROUGHPY_ALGEBRA_EXPORT
UnspecifiedAlgebraType construct_dense_algebra(
        scalars::ScalarArray&& data, const context_pointer& ctx,
        AlgebraType atype
);

ROUGHPY_ALGEBRA_EXPORT void check_contexts_compatible(
        const context_pointer& ref, const context_pointer& other
);
}// namespace dtl

}// namespace algebra
}// namespace rpy

RPY_WARNING_POP

#endif// ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_
