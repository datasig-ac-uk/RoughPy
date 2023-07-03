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

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_

#include "algebra_fwd.h"

#include <memory>
#include <stdexcept>
#include <type_traits>

#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>

#include "algebra_iterator.h"
#include "context_fwd.h"
#include "fallback_operations.h"
#include "basis.h"



namespace rpy {
namespace algebra {

namespace dtl {

/**
 * @brief Tag for all algebra interfaces
 *
 * Used to identify whether a class is an algebra interface or not in
 * various places. If you define an algebra interface, it should derive
 * from the base AlgebraInterface, which publicly derives from this tag.
 */
class ROUGHPY_ALGEBRA_EXPORT AlgebraInterfaceBase {
protected:
    context_pointer p_ctx;
    const scalars::ScalarType *p_coeff_type;
    VectorType m_vector_type;
    ImplementationType m_impl_type;

    explicit AlgebraInterfaceBase(context_pointer&& ctx,
                                 VectorType vtype,
                                 const scalars::ScalarType* stype,
                                 ImplementationType impl_type);


public:

    virtual ~AlgebraInterfaceBase();

    RPY_NO_DISCARD
    const context_pointer &context() const noexcept { return p_ctx; }
    RPY_NO_DISCARD
    ImplementationType impl_type() const noexcept { return m_impl_type; }
    RPY_NO_DISCARD
    VectorType storage_type() const noexcept { return m_vector_type; };
    RPY_NO_DISCARD
    const scalars::ScalarType *coeff_type() const noexcept { return p_coeff_type; };
};




template <typename Algebra, typename BasisType>
class AlgebraBasicProperties : public AlgebraInterfaceBase {
protected:

    BasisType m_basis;

    AlgebraBasicProperties(
        context_pointer&& ctx,
        VectorType vtype,
        const scalars::ScalarType* stype,
        ImplementationType impl_type
        )
        : AlgebraInterfaceBase(std::move(ctx), vtype, stype, impl_type),
          m_basis(basis_setup_helper<Algebra>::get(*context()))
    {}

public:
    using algebra_t = Algebra;
    using basis_t = BasisType;
    using id_t = uintptr_t;


//    virtual ~AlgebraBasicProperties() = default;

    // Type information
    RPY_NO_DISCARD
    id_t id() const noexcept;
    RPY_NO_DISCARD
    const BasisType& basis() const noexcept { return m_basis; }

    // Basic properties
    RPY_NO_DISCARD
    virtual dimn_t dimension() const = 0;
    RPY_NO_DISCARD
    virtual dimn_t size() const = 0;
    RPY_NO_DISCARD
    virtual bool is_zero() const = 0;
    RPY_NO_DISCARD
    virtual optional<deg_t> degree() const;
    RPY_NO_DISCARD
    virtual optional<deg_t> width() const;
    RPY_NO_DISCARD
    virtual optional<deg_t> depth() const;

    // Clone
    RPY_NO_DISCARD
    virtual Algebra clone() const = 0;
    RPY_NO_DISCARD
    virtual Algebra zero_like() const = 0;

    // Borrow
    RPY_NO_DISCARD
    virtual Algebra borrow() const = 0;
    RPY_NO_DISCARD
    virtual Algebra borrow_mut() = 0;

    virtual void clear() = 0;
    virtual void assign(const Algebra &other) = 0;

    // Display
    virtual std::ostream &print(std::ostream &os) const = 0;

    // Equality testing
    RPY_NO_DISCARD
    virtual bool equals(const Algebra &other) const = 0;
};


template <typename Base>
class AlgebraElementAccess : public Base {
protected:
    using Base::Base;

public:
    using typename Base::algebra_t;
    using typename Base::basis_t;


    using key_type = typename basis_t::key_type;

    // Element access
    virtual scalars::Scalar get(key_type key) const = 0;
    virtual scalars::Scalar get_mut(key_type key) = 0;
};


template <typename Base>
class AlgebraIteratorMethods : public Base{
protected:
    using Base::Base;

public:
    using typename Base::algebra_t;


    using const_iterator = AlgebraIterator<algebra_t>;

    // Iteration
    virtual const_iterator begin() const = 0;
    virtual const_iterator end() const = 0;

    virtual optional<scalars::ScalarArray> dense_data() const;
};


template <typename Base>
class AlgebraArithmetic : public Base {
protected:
    using Base::Base;

public:
    using typename Base::algebra_t;

    // Arithmetic
    virtual algebra_t uminus() const = 0;
    virtual algebra_t add(const algebra_t &other) const;
    virtual algebra_t sub(const algebra_t &other) const;
    virtual algebra_t mul(const algebra_t &other) const;
    virtual algebra_t smul(const scalars::Scalar &other) const;
    virtual algebra_t sdiv(const scalars::Scalar &other) const;

    // Inplace arithmetic
    virtual void add_inplace(const algebra_t &other) = 0;
    virtual void sub_inplace(const algebra_t &other) = 0;
    virtual void mul_inplace(const algebra_t &other) = 0;
    virtual void smul_inplace(const scalars::Scalar &other) = 0;
    virtual void sdiv_inplace(const scalars::Scalar &other) = 0;

    // Hybrid inplace arithmetic
    virtual void add_scal_mul(const algebra_t &rhs, const scalars::Scalar &scalar);
    virtual void sub_scal_mul(const algebra_t &rhs, const scalars::Scalar &scalar);
    virtual void add_scal_div(const algebra_t &rhs, const scalars::Scalar &scalar);
    virtual void sub_scal_div(const algebra_t &rhs, const scalars::Scalar &scalar);

    virtual void add_mul(const algebra_t &lhs, const algebra_t &rhs);
    virtual void sub_mul(const algebra_t &lhs, const algebra_t &rhs);
    virtual void mul_smul(const algebra_t &rhs, const scalars::Scalar &scalar);
    virtual void mul_sdiv(const algebra_t &rhs, const scalars::Scalar &scalar);
};


template <typename A, typename B, template <typename> class... Bases>
struct algebra_base_resolution;

template <typename A, typename B, template <typename> class First,
          template <typename> class... Remaining>
struct algebra_base_resolution<A, B, First, Remaining...> {
    using type = First<typename algebra_base_resolution<A, B, Remaining...>::type>;
};

template <typename A, typename B, template <typename> class Base>
struct algebra_base_resolution<A, B, Base> {
    using type = Base<AlgebraBasicProperties<A, B>>;
};

template <typename A, typename B>
struct algebra_base_resolution<A, B> {
    using type = AlgebraBasicProperties<A, B>;
};

}// namespace dtl


/**
 * @brief Base interface for algebra types
 * @tparam Algebra The externally facing algebra that this interface will
 * be used in.
 */
template <typename Algebra, typename BasisType>
class AlgebraInterface :
    public dtl::algebra_base_resolution<Algebra, BasisType,
                                        dtl::AlgebraArithmetic,
                                        dtl::AlgebraIteratorMethods,
                                        dtl::AlgebraElementAccess>::type
{
    using base_type = typename dtl::algebra_base_resolution<Algebra, BasisType,
                                                            dtl::AlgebraArithmetic,
                                                            dtl::AlgebraIteratorMethods,
                                                            dtl::AlgebraElementAccess>::type;
public:

    using algebra_interface_t = AlgebraInterface;
    using typename base_type::algebra_t;

protected:
    using base_type::base_type;
};


// Forward declarations of implementation templates
template <typename>
class OwnedStorageModel;
template <typename>
class BorrowedStorageModel;
template <typename, typename, template <typename> class>
class AlgebraImplementation;

namespace dtl {
template <typename Impl, template <typename, template <typename> class> class Wrapper>
using select_owned_or_borrowed_t = conditional_t<is_pointer<remove_reference_t<Impl>>::value,
                                                         Wrapper<remove_cv_t<remove_pointer_t<Impl>>, BorrowedStorageModel>,
                                                         Wrapper<remove_cv_ref_t<Impl>, OwnedStorageModel>>;

template <typename IFace>
struct with_interface {
    template <typename Impl, template <typename> class StorageModel>
    using type = AlgebraImplementation<IFace, Impl, StorageModel>;
};

ROUGHPY_ALGEBRA_EXPORT void print_empty_algebra(std::ostream &os);
ROUGHPY_ALGEBRA_EXPORT const scalars::ScalarType *context_to_scalars(const context_pointer &ptr);

}// namespace dtl

/**
 * @brief Base wrapper for algebra types
 * @tparam Interface Interface of algebra type
 * @tparam DerivedImpl Optional specialised template wrapper, use if
 *  AlgebraImplementation is not sufficient for the algebra interface.
 */
template <typename Interface, template <typename, template <typename> class> class DerivedImpl = dtl::with_interface<Interface>::template type>
class AlgebraBase {

protected:

    std::unique_ptr<Interface> p_impl;
    friend class algebra_access<Interface>;

    friend class algebra_access<typename Interface::algebra_interface_t>;
public:

    explicit AlgebraBase(std::unique_ptr<Interface> impl)
        : p_impl(std::move(impl)) {}
    explicit AlgebraBase(Interface *impl)
        : p_impl(impl) {}

    using interface_t = Interface;

    using basis_type = typename interface_t::basis_t;
    using key_type = typename interface_t::key_type;

    using algebra_t = typename Interface::algebra_t;
    using const_iterator = AlgebraIterator<algebra_t>;

    AlgebraBase() : p_impl(nullptr) {}
    AlgebraBase(const AlgebraBase &other);
    AlgebraBase(AlgebraBase &&other) noexcept;

    AlgebraBase &operator=(const AlgebraBase &other);
    AlgebraBase &operator=(AlgebraBase &&other) noexcept;

    explicit AlgebraBase(context_pointer ctx);

    template <typename Impl,
              typename = enable_if_t<!is_same<remove_cv_ref_t<Impl>, algebra_t>::value>>
    explicit AlgebraBase(context_pointer ctx, Impl &&arg)
        : p_impl(new dtl::select_owned_or_borrowed_t<Impl, DerivedImpl>(std::move(ctx), std::forward<Impl>(arg))) {}

    template <typename Impl, typename... Args>
    static enable_if_t<!is_base_of<Interface, Impl>::value, algebra_t>
    from_args(context_pointer ctx, Args &&...args) {
        return algebra_t(std::move(ctx), new dtl::select_owned_or_borrowed_t<Impl, DerivedImpl>(std::forward<Args>(args)...));
    }

    template <typename Wrapper, typename... Args>
    static enable_if_t<is_base_of<Interface, Wrapper>::value, algebra_t>
    from_args(context_pointer ctx, Args &&...args) {
        return algebra_t(std::move(ctx), new Wrapper(std::forward<Args>(args)...));
    }

    RPY_NO_DISCARD
    algebra_t borrow() const;
    RPY_NO_DISCARD
    algebra_t borrow_mut();

    RPY_NO_DISCARD
    const Interface &operator*() const noexcept { return *p_impl; }
    RPY_NO_DISCARD
    Interface &operator*() noexcept { return *p_impl; }
    RPY_NO_DISCARD
    const Interface *operator->() const noexcept { return p_impl.get(); }
    RPY_NO_DISCARD
    Interface *operator->() noexcept { return p_impl.get(); }

    RPY_NO_DISCARD
    constexpr operator bool() const noexcept { return static_cast<bool>(p_impl); }

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
    const scalars::ScalarType *coeff_type() const noexcept;

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
    static bool is_equivalent_to_zero(const AlgebraBase &arg) {
        // For the moment, we just check if the arg has a null-p_impl
        // In the future we might do something else.
        return arg.p_impl == nullptr;
    }

    RPY_NO_DISCARD
    static algebra_t &downcast(AlgebraBase &arg) { return static_cast<algebra_t &>(arg); }
    RPY_NO_DISCARD
    static const algebra_t &downcast(const AlgebraBase &arg) { return static_cast<const algebra_t &>(arg); }

public:
    RPY_NO_DISCARD
    algebra_t uminus() const;
    RPY_NO_DISCARD
    algebra_t add(const algebra_t &rhs) const;
    RPY_NO_DISCARD
    algebra_t sub(const algebra_t &rhs) const;
    RPY_NO_DISCARD
    algebra_t mul(const algebra_t &rhs) const;
    RPY_NO_DISCARD
    algebra_t smul(const scalars::Scalar &rhs) const;
    RPY_NO_DISCARD
    algebra_t sdiv(const scalars::Scalar &rhs) const;

    RPY_NO_DISCARD
    algebra_t &add_inplace(const algebra_t &rhs);
    RPY_NO_DISCARD
    algebra_t &sub_inplace(const algebra_t &rhs);
    RPY_NO_DISCARD
    algebra_t &mul_inplace(const algebra_t &rhs);
    RPY_NO_DISCARD
    algebra_t &smul_inplace(const scalars::Scalar &rhs);
    RPY_NO_DISCARD
    algebra_t &sdiv_inplace(const scalars::Scalar &rhs);

    RPY_NO_DISCARD
    algebra_t &add_scal_mul(const algebra_t &lhs, const scalars::Scalar &rhs);
    RPY_NO_DISCARD
    algebra_t &sub_scal_mul(const algebra_t &lhs, const scalars::Scalar &rhs);
    RPY_NO_DISCARD
    algebra_t &add_scal_div(const algebra_t &lhs, const scalars::Scalar &rhs);
    RPY_NO_DISCARD
    algebra_t &sub_scal_div(const algebra_t &lhs, const scalars::Scalar &rhs);

    RPY_NO_DISCARD
    algebra_t &add_mul(const algebra_t &lhs, const algebra_t &rhs);
    RPY_NO_DISCARD
    algebra_t &sub_mul(const algebra_t &lhs, const algebra_t &rhs);
    RPY_NO_DISCARD
    algebra_t &mul_smul(const algebra_t &lhs, const scalars::Scalar &rhs);
    RPY_NO_DISCARD
    algebra_t &mul_sdiv(const algebra_t &lhs, const scalars::Scalar &rhs);

    std::ostream &print(std::ostream &os) const;

    RPY_NO_DISCARD
    bool operator==(const algebra_t &other) const;
    RPY_NO_DISCARD
    bool operator!=(const algebra_t &other) const { return !operator==(other); }

private:
    RPY_SERIAL_ACCESS();
    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();
};

// Definitions of all the member functions

template <typename Algebra, typename BasisType>
typename dtl::AlgebraBasicProperties<Algebra, BasisType>::id_t dtl::AlgebraBasicProperties<Algebra, BasisType>::id() const noexcept {
    return 0;
}
template <typename Algebra, typename BasisType>
optional<deg_t> dtl::AlgebraBasicProperties<Algebra, BasisType>::degree() const {
    return optional<deg_t>();
}
template <typename Algebra, typename BasisType>
optional<deg_t> dtl::AlgebraBasicProperties<Algebra, BasisType>::width() const {
    return optional<deg_t>();
}
template <typename Algebra, typename BasisType>
optional<deg_t> dtl::AlgebraBasicProperties<Algebra, BasisType>::depth() const {
    return optional<deg_t>();
}

template <typename Base>
optional<scalars::ScalarArray> dtl::AlgebraIteratorMethods<Base>::dense_data() const {
    return {};
}

template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t dtl::AlgebraArithmetic<Base>::add(const algebra_t &other) const {
    auto result = this->clone();
    result->add_inplace(other);
    return result;
}
template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t dtl::AlgebraArithmetic<Base>::sub(const algebra_t &other) const {
    auto result = this->clone();
    result->sub_inplace(other);
    return result;
}
template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t dtl::AlgebraArithmetic<Base>::mul(const algebra_t &other) const {
    auto result = this->clone();
    result->mul_inplace(other);
    return result;
}
template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t dtl::AlgebraArithmetic<Base>::smul(const scalars::Scalar &other) const {
    auto result = this->clone();
    result->smul_inplace(other);
    return result;
}
template <typename Base>
typename dtl::AlgebraArithmetic<Base>::algebra_t dtl::AlgebraArithmetic<Base>::sdiv(const scalars::Scalar &other) const {
    auto result = this->clone();
    result->sdiv_inplace(other);
    return result;
}

template <typename Base>
void dtl::AlgebraArithmetic<Base>::add_scal_mul(const algebra_t &rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.smul(scalar);
    add_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::sub_scal_mul(const algebra_t &rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.smul(scalar);
    sub_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::add_scal_div(const algebra_t &rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.sdiv(scalar);
    add_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::sub_scal_div(const algebra_t &rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.sdiv(scalar);
    sub_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::add_mul(const algebra_t &lhs, const algebra_t &rhs) {
    auto tmp = lhs.mul(rhs);
    add_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::sub_mul(const algebra_t &lhs, const algebra_t &rhs) {
    auto tmp = lhs.mul(rhs);
    sub_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::mul_smul(const algebra_t &rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.smul(scalar);
    mul_inplace(tmp);
}
template <typename Base>
void dtl::AlgebraArithmetic<Base>::mul_sdiv(const algebra_t &rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.sdiv(scalar);
    mul_inplace(tmp);
}


namespace dtl {

ROUGHPY_ALGEBRA_EXPORT
UnspecifiedAlgebraType try_create_new_empty(context_pointer ctx, AlgebraType alg_type);

template <typename Interface>
std::unique_ptr<Interface> downcast_interface_ptr(UnspecifiedAlgebraType ptr) {
    return std::unique_ptr<Interface>(
        reinterpret_cast<Interface*>(ptr.release())
        );
}


}// namespace dtl


#define RPY_CHECK_CONTEXTS(OTHER) \
    RPY_CHECK(context()->check_compatible(*(OTHER).context()))

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>::AlgebraBase(context_pointer ctx)
    : p_impl(nullptr) {
    /*
     * Try and create a new empty instance by appealing to the context
     * and passing in the type to be created. This will return either
     * a pointer to a AlgebraInterfaceTag, which will necessarily actually
     * point to an object of type Interface, or a null pointer if this
     * construction is not possible.
     */

    p_impl = dtl::downcast_interface_ptr<Interface>(
        dtl::try_create_new_empty(std::move(ctx), algebra_t::s_alg_type));
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>::AlgebraBase(const AlgebraBase &other) {
    if (other.p_impl) {
        *this = other.p_impl->clone();
    }
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>::AlgebraBase(AlgebraBase &&other) noexcept
    : p_impl(std::move(other.p_impl)) {
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl> &AlgebraBase<Interface, DerivedImpl>::operator=(const AlgebraBase &other) {
    if (&other != this) {
        if (other.p_impl) {
            *this = other.p_impl->clone();
        } else {
            p_impl.reset();
        }
    }
    return *this;
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl> &AlgebraBase<Interface, DerivedImpl>::operator=(AlgebraBase &&other) noexcept {
    if (&other != this) {
        p_impl = std::move(other.p_impl);
    }
    return *this;
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
context_pointer AlgebraBase<Interface, DerivedImpl>::context() const noexcept {
    return (p_impl) ? p_impl->context() : nullptr;
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::borrow() const {
    if (p_impl) {
        return p_impl->borrow();
    }
    return algebra_t();
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::borrow_mut() {
    if (p_impl) {
        return p_impl->borrow_mut();
    }
    return algebra_t();
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::add(const algebra_t &rhs) const {
    if (is_equivalent_to_zero(rhs)) {
        if (is_equivalent_to_zero(*this)) {
            return algebra_t();
        }
        return p_impl->clone();
    }
    RPY_CHECK_CONTEXTS(rhs);

    if (is_equivalent_to_zero(*this)) {
        return rhs.p_impl->clone();
    }
    return p_impl->add(rhs);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::sub(const algebra_t &rhs) const {
    if (is_equivalent_to_zero(rhs)) {
        if (is_equivalent_to_zero(*this)) {
            return algebra_t();
        }
        return p_impl->clone();
    }

    RPY_CHECK_CONTEXTS(rhs);

    if (is_equivalent_to_zero(*this)) {
        return rhs.p_impl->uminus();
    }
    return p_impl->sub(rhs);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::mul(const algebra_t &rhs) const {
    if (is_equivalent_to_zero(rhs)) {
        if (is_equivalent_to_zero(*this)) {
            return algebra_t();
        }
        return p_impl->clone();
    }

    RPY_CHECK_CONTEXTS(rhs);

    if (is_equivalent_to_zero(*this)) {
        return rhs.p_impl->clone();
    }
    return p_impl->mul(rhs);
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
dimn_t AlgebraBase<Interface, DerivedImpl>::dimension() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->dimension();
    }
    return 0;
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
dimn_t AlgebraBase<Interface, DerivedImpl>::size() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->size();
    }
    return 0;
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
bool AlgebraBase<Interface, DerivedImpl>::is_zero() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->is_zero();
    }
    return true;
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBase<Interface, DerivedImpl>::width() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->width();
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBase<Interface, DerivedImpl>::depth() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->depth();
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
optional<deg_t> AlgebraBase<Interface, DerivedImpl>::degree() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->degree();
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
VectorType AlgebraBase<Interface, DerivedImpl>::storage_type() const noexcept {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->storage_type();
    }
    return VectorType::Sparse;
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
const scalars::ScalarType *AlgebraBase<Interface, DerivedImpl>::coeff_type() const noexcept {
    if (p_impl) {
        return p_impl->coeff_type();
    }
    return nullptr;
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
scalars::Scalar AlgebraBase<Interface, DerivedImpl>::operator[](key_type k) const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->get(k);
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
scalars::Scalar AlgebraBase<Interface, DerivedImpl>::operator[](key_type k) {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->get_mut(k);
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBase<Interface, DerivedImpl>::const_iterator
AlgebraBase<Interface, DerivedImpl>::begin() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->begin();
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename AlgebraBase<Interface, DerivedImpl>::const_iterator
AlgebraBase<Interface, DerivedImpl>::end() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->end();
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
optional<rpy::scalars::ScalarArray> AlgebraBase<Interface, DerivedImpl>::dense_data() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->dense_data();
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::uminus() const {
    if (is_equivalent_to_zero(*this)) {
        return algebra_t();
    }
    return p_impl->uminus();
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::smul(const scalars::Scalar &rhs) const {
    if (is_equivalent_to_zero(*this)) {
        return algebra_t();
    }
    if (rhs.is_zero()) {
        return p_impl->zero_like();
    }
    // The implementation should perform the necessary scalar casting
    return p_impl->smul(rhs);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::sdiv(const scalars::Scalar &rhs) const {
    if (is_equivalent_to_zero(*this)) {
        return algebra_t();
    }
    if (rhs.is_zero()) {
        throw std::invalid_argument("cannot divide by zero");
    }
    // The implementation should perform the necessary scalar casting
    return p_impl->sdiv(rhs);
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::add_inplace(const algebra_t &rhs) {
    if (is_equivalent_to_zero(rhs)) {
        return downcast(*this);
    }
    if (is_equivalent_to_zero(*this)) {
        *this = algebra_t();
    }
    RPY_CHECK_CONTEXTS(rhs);
    p_impl->add_inplace(rhs);
    return downcast(*this);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::sub_inplace(const algebra_t &rhs) {
    if (is_equivalent_to_zero(rhs)) {
        return downcast(*this);
    }
    if (is_equivalent_to_zero(*this)) {
        *this = algebra_t();
    }
    RPY_CHECK_CONTEXTS(rhs);
    p_impl->sub_inplace(rhs);
    return downcast(*this);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::mul_inplace(const algebra_t &rhs) {
    if (is_equivalent_to_zero(rhs)) {
        return downcast(*this);
    }
    if (is_equivalent_to_zero(*this)) {
        *this = algebra_t();
    }
    RPY_CHECK_CONTEXTS(rhs);
    p_impl->mul_inplace(rhs);
    return downcast(*this);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::smul_inplace(const scalars::Scalar &rhs) {
    if (!is_equivalent_to_zero(*this)) {
        p_impl->smul_inplace(rhs);
    }
    return downcast(*this);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::sdiv_inplace(const scalars::Scalar &rhs) {
    if (!is_equivalent_to_zero(*this)) {
        if (rhs.is_zero()) {
            throw std::invalid_argument("cannot divide by zero");
        }
        p_impl->sdiv_inplace(rhs);
    }
    return downcast(*this);
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::add_scal_mul(const algebra_t &lhs, const scalars::Scalar &rhs) {
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
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::sub_scal_mul(const algebra_t &lhs, const scalars::Scalar &rhs) {
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
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::add_scal_div(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (!is_equivalent_to_zero(lhs)) {
        RPY_CHECK_CONTEXTS(lhs);

        if (rhs.is_zero()) {
            throw std::invalid_argument("cannot divide by zero");
        }
        if (!is_equivalent_to_zero(*this)) {
            p_impl->add_scal_div(lhs, rhs);
        } else {
            *this = lhs.sdiv(rhs);
        }
    }
    return downcast(*this);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::sub_scal_div(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (!is_equivalent_to_zero(lhs)) {
        RPY_CHECK_CONTEXTS(lhs);

        if (rhs.is_zero()) {
            throw std::invalid_argument("cannot divide by zero");
        }
        if (!is_equivalent_to_zero(*this)) {
            p_impl->sub_scal_div(lhs, rhs);
        } else {
            *this = lhs.sdiv(rhs).uminus();
        }
    }
    return downcast(*this);
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::add_mul(const algebra_t &lhs, const algebra_t &rhs) {
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
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::sub_mul(const algebra_t &lhs, const algebra_t &rhs) {
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
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::mul_smul(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (!is_equivalent_to_zero(lhs) && !rhs.is_zero()) {
        RPY_CHECK_CONTEXTS(lhs);

        if (!is_equivalent_to_zero(*this)) {
            p_impl->mul_smul(lhs, rhs);
        }
    } else if (!is_equivalent_to_zero(*this)) {
        p_impl->clear();
    }
    return downcast(*this);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t &AlgebraBase<Interface, DerivedImpl>::mul_sdiv(const algebra_t &lhs, const scalars::Scalar &rhs) {
    if (!is_equivalent_to_zero(lhs)) {
        RPY_CHECK_CONTEXTS(lhs);

        if (rhs.is_zero()) {
            throw std::invalid_argument("cannot divide by zero");
        }
        if (!is_equivalent_to_zero(*this)) {
            p_impl->mul_sdiv(lhs, rhs);
        }
    } else if (!is_equivalent_to_zero(*this)) {
        p_impl->clear();
    }
    return downcast(*this);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
bool AlgebraBase<Interface, DerivedImpl>::operator==(const algebra_t &other) const {
    if (is_equivalent_to_zero(*this)) {
        return is_equivalent_to_zero(other) || other->is_zero();
    }
    if (is_equivalent_to_zero(other)) {
        return p_impl->is_zero();
    }

    if (!context()->check_compatible(*other.context())) {
        return false;
    }

    return p_impl->equals(other);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
std::ostream &AlgebraBase<Interface, DerivedImpl>::print(std::ostream &os) const {
    if (is_equivalent_to_zero(*this)) {
        dtl::print_empty_algebra(os);
    } else {
        p_impl->print(os);
    }
    return os;
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
inline std::ostream &operator<<(std::ostream &os, const AlgebraBase<Interface, DerivedImpl> &alg) {
    return alg.print(os);
}

#undef RPY_CHECK_CONTEXTS

template <typename Interface, template <typename, template <typename> class> class Derived>
template <typename Archive>
void AlgebraBase<Interface, Derived>::save(Archive &archive, const std::uint32_t RPY_UNUSED_VAR version) const {
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
                RPY_SERIAL_SERIALIZE_BARE(std::make_pair(item.key(), item
                                                                         .value()));
            }
        }

    }

}

namespace dtl {

ROUGHPY_ALGEBRA_EXPORT
UnspecifiedAlgebraType construct_dense_algebra(scalars::ScalarArray&& data, const context_pointer& ctx, AlgebraType atype);


}

template <typename Interface, template <typename, template <typename> class> class Derived>
template <typename Archive>
void AlgebraBase<Interface, Derived>::load(Archive& archive, const std::uint32_t RPY_UNUSED_VAR version) {
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

    if (!has_values) {
        return;
    }

    if (vtype == VectorType::Dense) {

        scalars::ScalarArray tmp;
        RPY_SERIAL_SERIALIZE_NVP("dense_data", tmp);
        p_impl = dtl::downcast_interface_ptr<Interface>(dtl::construct_dense_algebra(std::move(tmp), ctx, atype));
    } else {
        p_impl = dtl::downcast_interface_ptr<Interface>(dtl::try_create_new_empty(ctx, atype));

        serial::size_type size;
        RPY_SERIAL_SERIALIZE_SIZE(size);

        for (serial::size_type i=0; i<size; ++i) {
            std::pair<typename Interface::key_type, scalars::Scalar> val;
            RPY_SERIAL_SERIALIZE_BARE(val);
            p_impl->get_mut(val.first) = val.second;
        }
    }
}





}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_
