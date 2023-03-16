#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_

#include "algebra_fwd.h"

#include <memory>
#include <stdexcept>
#include <type_traits>

#include <roughpy/config/implementation_types.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>

#include "algebra_iterator.h"
#include "context_fwd.h"
#include "fallback_operations.h"

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
struct AlgebraInterfaceTag {};

}// namespace dtl


using UnspecifiedAlgebraType = dtl::AlgebraInterfaceTag*;

/**
 * @brief Base interface for algebra types
 * @tparam Algebra The externally facing algebra that this interface will
 * be used in.
 */
template <typename Algebra>
class AlgebraInterface : public dtl::AlgebraInterfaceTag {

protected:
    context_pointer p_ctx;
    VectorType m_vector_type;
    const scalars::ScalarType *p_coeff_type;
    ImplementationType m_impl_type;

    explicit AlgebraInterface(context_pointer&& ctx,
                              VectorType vtype,
                              const scalars::ScalarType *stype,
                              ImplementationType impl_type)
        : p_ctx(ctx), m_vector_type(vtype), p_coeff_type(stype), m_impl_type(impl_type) {}


public:
    using algebra_t = Algebra;
    using const_iterator = AlgebraIterator;
    using algebra_interface_t = AlgebraInterface;
    using id_t = std::uintptr_t;


public:

    virtual ~AlgebraInterface() = default;

    // Type information
    id_t id() const noexcept;
    const context_pointer& context() const noexcept { return p_ctx; }
    ImplementationType impl_type() const noexcept { return m_impl_type; }
    VectorType storage_type() const noexcept { return m_vector_type; };
    const scalars::ScalarType *coeff_type() const noexcept { return p_coeff_type; };

    // Basic properties
    virtual dimn_t dimension() const = 0;
    virtual dimn_t size() const = 0;
    virtual bool is_zero() const = 0;
    virtual optional<deg_t> degree() const;
    virtual optional<deg_t> width() const;
    virtual optional<deg_t> depth() const;

    // Clone
    virtual Algebra clone() const = 0;
    virtual Algebra zero_like() const = 0;

//    virtual Algebra convert_from_iterator(AlgebraIterator begin, AlgebraIterator end) const = 0;

    // Element access
    virtual scalars::Scalar get(key_type key) const = 0;
    virtual scalars::Scalar get_mut(key_type key) = 0;

    // Iteration
    virtual AlgebraIterator begin() const = 0;
    virtual AlgebraIterator end() const = 0;

    virtual optional<scalars::ScalarArray> dense_data() const;

    virtual void clear() = 0;
    virtual void assign(const Algebra &other) = 0;

    // Arithmetic
    virtual Algebra uminus() const = 0;
    virtual Algebra add(const Algebra &other) const;
    virtual Algebra sub(const Algebra &other) const;
    virtual Algebra mul(const Algebra &other) const;
    virtual Algebra smul(const scalars::Scalar &other) const;
    virtual Algebra sdiv(const scalars::Scalar &other) const;

    // Inplace arithmetic
    virtual void add_inplace(const Algebra& other) = 0;
    virtual void sub_inplace(const Algebra& other) = 0;
    virtual void mul_inplace(const Algebra& other) = 0;
    virtual void smul_inplace(const scalars::Scalar &other) = 0;
    virtual void sdiv_inplace(const scalars::Scalar &other) = 0;

    // Hybrid inplace arithmetic
    virtual void add_scal_mul(const Algebra&rhs, const scalars::Scalar &scalar);
    virtual void sub_scal_mul(const Algebra&rhs, const scalars::Scalar &scalar);
    virtual void add_scal_div(const Algebra&rhs, const scalars::Scalar &scalar);
    virtual void sub_scal_div(const Algebra&rhs, const scalars::Scalar &scalar);

    virtual void add_mul(const Algebra&lhs, const Algebra&rhs);
    virtual void sub_mul(const Algebra&lhs, const Algebra&rhs);
    virtual void mul_smul(const Algebra&rhs, const scalars::Scalar &scalar);
    virtual void mul_sdiv(const Algebra&rhs, const scalars::Scalar &scalar);

    // Display
    virtual std::ostream &print(std::ostream &os) const = 0;

    // Equality testing
    virtual bool equals(const Algebra &other) const = 0;
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
using select_owned_or_borrowed_t = traits::conditional_t<traits::is_pointer<traits::remove_reference_t<Impl>>::value,
                                                      Wrapper<traits::remove_cv_t<traits::remove_pointer_t<Impl>>, BorrowedStorageModel>,
                                                      Wrapper<traits::remove_cv_ref_t<Impl>, OwnedStorageModel>>;

template <typename IFace>
struct with_interface {
    template <typename Impl, template <typename> class StorageModel>
    using type = AlgebraImplementation<IFace, Impl, StorageModel>;
};

ROUGHPY_ALGEBRA_EXPORT void print_empty_algebra(std::ostream &os);
ROUGHPY_ALGEBRA_EXPORT const scalars::ScalarType* context_to_scalars(const context_pointer& ptr);

}// namespace dtl

/**
 * @brief Base wrapper for algebra types
 * @tparam Interface Interface of algebra type
 * @tparam DerivedImpl Optional specialised template wrapper, use if
 *  AlgebraImplementation is not sufficient for the algebra interface.
 */
template <typename Interface, template <typename, template <typename> class> class DerivedImpl = dtl::with_interface<Interface>::template type>
class AlgebraBase {

    explicit AlgebraBase(std::unique_ptr<Interface> impl)
        : p_impl(std::move(impl)) {}
    explicit AlgebraBase(Interface *impl)
        : p_impl(impl) {}

protected:
    std::unique_ptr<Interface> p_impl;

    friend class algebra_access<Interface>;
    friend class algebra_access<typename Interface::algebra_interface_t>;

public:
    using interface_t = Interface;
    using algebra_t = typename Interface::algebra_t;
    using const_iterator = AlgebraIterator;


public:
    AlgebraBase() : p_impl(nullptr) {}
    AlgebraBase(const AlgebraBase& other);
    AlgebraBase(AlgebraBase&& other) noexcept;

    AlgebraBase& operator=(const AlgebraBase& other);
    AlgebraBase& operator=(AlgebraBase&& other) noexcept;

    explicit AlgebraBase(context_pointer ctx);

    template <typename Impl,
              typename = traits::enable_if_t<!traits::is_same<traits::remove_cv_ref_t<Impl>, algebra_t>::value>>
    explicit AlgebraBase(context_pointer ctx, Impl &&arg)
        : p_impl(new dtl::select_owned_or_borrowed_t<Impl, DerivedImpl>(std::move(ctx), std::forward<Impl>(arg))) {}

    template <typename Impl, typename... Args>
    static traits::enable_if_t<!traits::is_base_of<Interface, Impl>::value, algebra_t>
    from_args(context_pointer ctx, Args &&...args) {
        return algebra_t(std::move(ctx), new dtl::select_owned_or_borrowed_t<Impl, DerivedImpl>(std::forward<Args>(args)...));
    }

    template <typename Wrapper, typename... Args>
    static traits::enable_if_t<traits::is_base_of<Interface, Wrapper>::value, algebra_t>
    from_args(context_pointer ctx, Args &&...args) {
        return algebra_t(std::move(ctx), new Wrapper(std::forward<Args>(args)...));
    }

    const Interface &operator*() const noexcept { return *p_impl; }
    Interface &operator*() noexcept { return *p_impl; }
    const Interface *operator->() const noexcept { return p_impl.get(); }
    Interface *operator->() noexcept { return p_impl.get(); }

    constexpr operator bool () const noexcept { return static_cast<bool>(p_impl); }

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

    const_iterator begin() const;
    const_iterator end() const;

    optional<scalars::ScalarArray> dense_data() const;

protected:
    static bool is_equivalent_to_zero(const AlgebraBase &arg) {
        // For the moment, we just check if the arg has a null-p_impl
        // In the future we might do something else.
        return arg.p_impl == nullptr;
    }

    static algebra_t &downcast(AlgebraBase &arg) { return static_cast<algebra_t &>(arg); }
    static const algebra_t &downcast(const AlgebraBase &arg) { return static_cast<const algebra_t &>(arg); }

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
};

// Definitions of all the member functions

template <typename Algebra>
typename AlgebraInterface<Algebra>::id_t AlgebraInterface<Algebra>::id() const noexcept {
    return {};
}



template <typename Algebra>
optional<deg_t> AlgebraInterface<Algebra>::degree() const {
    return {};
}
template <typename Algebra>
optional<deg_t> AlgebraInterface<Algebra>::width() const {
    return {};
}
template <typename Algebra>
optional<deg_t> AlgebraInterface<Algebra>::depth() const {
    return {};
}
template <typename Algebra>
optional<scalars::ScalarArray> AlgebraInterface<Algebra>::dense_data() const {
    return {};
}
template <typename Algebra>
Algebra AlgebraInterface<Algebra>::add(const Algebra&other) const {
    auto result = clone();
    result->add_inplace(other);
    return result;
}
template <typename Algebra>
Algebra AlgebraInterface<Algebra>::sub(const Algebra&other) const {
    auto result = clone();
    result->sub_inplace(other);
    return result;
}
template <typename Algebra>
Algebra AlgebraInterface<Algebra>::mul(const Algebra&other) const {
    auto result = clone();
    result->mul_inplace(other);
    return result;
}
template <typename Algebra>
Algebra AlgebraInterface<Algebra>::smul(const scalars::Scalar &other) const {
    auto result = clone();
    result->smul_inplace(other);
    return result;
}
template <typename Algebra>
Algebra AlgebraInterface<Algebra>::sdiv(const scalars::Scalar &other) const {
    auto result = clone();
    result->sdiv_inplace(other);
    return result;
}
template <typename Algebra>
void AlgebraInterface<Algebra>::add_scal_mul(const Algebra&rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.smul(scalar);
    add_inplace(tmp);
}
template <typename Algebra>
void AlgebraInterface<Algebra>::sub_scal_mul(const Algebra&rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.smul(scalar);
    sub_inplace(tmp);
}
template <typename Algebra>
void AlgebraInterface<Algebra>::add_scal_div(const Algebra&rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.sdiv(scalar);
    add_inplace(tmp);
}
template <typename Algebra>
void AlgebraInterface<Algebra>::sub_scal_div(const Algebra&rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.sdiv(scalar);
    sub_inplace(tmp);
}
template <typename Algebra>
void AlgebraInterface<Algebra>::add_mul(const Algebra&lhs, const Algebra&rhs) {
    auto tmp = lhs.mul(rhs);
    add_inplace(tmp);
}
template <typename Algebra>
void AlgebraInterface<Algebra>::sub_mul(const Algebra&lhs, const Algebra&rhs) {
    auto tmp = lhs.mul(rhs);
    sub_inplace(tmp);
}
template <typename Algebra>
void AlgebraInterface<Algebra>::mul_smul(const Algebra&rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.smul(scalar);
    mul_inplace(tmp);
}
template <typename Algebra>
void AlgebraInterface<Algebra>::mul_sdiv(const Algebra&rhs, const scalars::Scalar &scalar) {
    auto tmp = rhs.sdiv(scalar);
    mul_inplace(tmp);
}

namespace dtl {

ROUGHPY_ALGEBRA_EXPORT
UnspecifiedAlgebraType try_create_new_empty(context_pointer ctx, AlgebraType alg_type);

}


template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
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
    p_impl = std::unique_ptr<Interface>(
        static_cast<Interface*>(dtl::try_create_new_empty(std::move(ctx), algebra_t::s_alg_type))
        );
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>::AlgebraBase(const AlgebraBase &other) {
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl>::AlgebraBase(AlgebraBase &&other) noexcept {
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl> &AlgebraBase<Interface, DerivedImpl>::operator=(const AlgebraBase &other) {
    return *this;
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraBase<Interface, DerivedImpl> &AlgebraBase<Interface, DerivedImpl>::operator=(AlgebraBase &&other) noexcept {
    return *this;
}

template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::add(const algebra_t &rhs) const {
    if (is_equivalent_to_zero(rhs)) {
        if (is_equivalent_to_zero(*this)) {
            return algebra_t();
        }
        return p_impl->clone();
    }
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
    if (is_equivalent_to_zero(*this)) {
        return rhs.p_impl->uminus();
    }
    return p_impl->sub(rhs);
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
typename Interface::algebra_t AlgebraBase<Interface, DerivedImpl>::mul(const algebra_t &rhs) const {
    if (!is_equivalent_to_zero(rhs)) {
        if (is_equivalent_to_zero(*this)) {
            return algebra_t();
        }
        return p_impl->clone();
    }
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
    if (is_equivalent_to_zero(*this)) {
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
    if (!is_equivalent_to_zero(*this)) {
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
AlgebraIterator AlgebraBase<Interface, DerivedImpl>::begin() const {
    if (!is_equivalent_to_zero(*this)) {
        return p_impl->begin();
    }
    return {};
}
template <typename Interface, template <typename, template <typename> class> class DerivedImpl>
AlgebraIterator AlgebraBase<Interface, DerivedImpl>::end() const {
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

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_BASE_H_
