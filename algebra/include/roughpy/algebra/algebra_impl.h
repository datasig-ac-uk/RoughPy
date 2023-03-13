#ifndef ROUGHPY_ALGEBRA_ALGEBRA_IMPL_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_IMPL_H_

#include <boost/type_traits/copy_cv.hpp>

#include <roughpy/config/traits.h>
#include <roughpy/config/helpers.h>
#include <roughpy/scalars/scalar_traits.h>

#include "algebra_base.h"
#include "algebra_info.h"
#include "algebra_iterator_impl.h"

namespace rpy {
namespace algebra {

template <typename Interface, typename Impl>
class ImplAccessLayer : public Interface {
protected:
    using Interface::Interface;

public:
    virtual Impl &get_data() noexcept = 0;
    virtual const Impl &get_data() const noexcept = 0;
};

template <typename Impl, typename Interface>
inline traits::copy_cv_t<Impl, Interface> &
algebra_cast(Interface &arg) noexcept {
    using access_t = traits::copy_cv_t<ImplAccessLayer<Interface, Impl>, Interface>;
    static_assert(traits::is_base_of<dtl::AlgebraInterfaceTag, Interface>::value,
                  "casting to algebra implementation is only possible for interfaces");
    assert(dynamic_cast<access_t*>(std::addressof(arg) == std::addressof(arg)));
    return static_cast<access_t&>(arg).get_data();
}

template <typename Impl>
class OwnedStorageModel {
    Impl m_data;

protected:
    static constexpr ImplementationType s_type = ImplementationType::Owned;

    template <typename... Args>
    explicit OwnedStorageModel(Args &&...args) : m_data(std::forward<Args>(args)...) {}
    explicit OwnedStorageModel(Impl &&arg) : m_data(std::move(arg)) {}
    explicit OwnedStorageModel(const Impl &arg) : m_data(arg) {}

    Impl &data() noexcept { return m_data; }
    const Impl &data() const noexcept { return m_data; }
};

template <typename Impl>
class BorrowedStorageModel {
    Impl *p_data;

protected:
    static constexpr ImplementationType s_type = ImplementationType::Borrowed;

    explicit BorrowedStorageModel(Impl *arg) : p_data(arg) {}

    Impl &data() noexcept { return *p_data; }
    const Impl &data() const noexcept { return *p_data; }
};

namespace dtl {

template <typename Impl>
class ConvertedArgument {
    const Impl &m_ref;
    optional<Impl> m_holder;

public:
    ConvertedArgument(Impl &&converted)
        : m_holder(std::move(converted)), m_ref(m_holder.get()) {}
    ConvertedArgument(const Impl &same_type)
        : m_holder(), m_ref(same_type) {}
    operator const Impl &() const noexcept {
        return m_ref;
    }
};

template <typename T>
using d_has_as_ptr_t = decltype(traits::declval<const T&>().base_vector().as_ptr());

#define RPY_HAS_FUSED_OP_CHECKER(NAME)          \
    template <typename T>                       \
    using d_##NAME = decltype(traits::declval<T&>().NAME())

RPY_HAS_FUSED_OP_CHECKER(add_scal_prod);
RPY_HAS_FUSED_OP_CHECKER(sub_scal_prod);
RPY_HAS_FUSED_OP_CHECKER(add_scal_div);
RPY_HAS_FUSED_OP_CHECKER(sub_scal_div);
RPY_HAS_FUSED_OP_CHECKER(add_mul);
RPY_HAS_FUSED_OP_CHECKER(sub_mul);
RPY_HAS_FUSED_OP_CHECKER(mul_scal_prod);
RPY_HAS_FUSED_OP_CHECKER(mul_scal_div);

#undef RPY_HAS_FUSED_OP_CHECKER

struct no_implementation {};
struct has_implementation {};

template <template <typename> class MF, typename T>
using use_impl_t = traits::conditional_t<
    traits::is_detected<MF, T>::value,
    has_implementation,
    no_implementation
>;

}// namespace dtl

template <typename Interface, typename Impl, template <typename> class StorageModel>
class AlgebraImplementation
    : protected StorageModel<Impl>,
      public ImplAccessLayer<Interface, Impl> {
    using storage_base_t = StorageModel<Impl>;
    using access_layer_t = ImplAccessLayer<Interface, Impl>;

    static_assert(traits::is_base_of<dtl::AlgebraInterfaceTag, Interface>::value,
                  "algebra_interface must be an accessible base of Interface");

public:
    using interface_t = Interface;
    using algebra_t = typename Interface::algebra_t;
    using algebra_interface_t = typename Interface::algebra_interface_t;

    static constexpr AlgebraType s_alg_type = algebra_t::s_alg_type;

    using scalar_type = typename algebra_info<Impl>::scalar_type;
    using rational_type = typename algebra_info<Impl>::rational_type;
    using basis_type = typename algebra_info<Impl>::basis_type;

protected:
    using storage_base_t::data;

public:
    Impl &get_data() noexcept override {
        return data();
    }
    const Impl &get_data() const noexcept override {
        return data();
    }

public:
    template <typename... Args>
    explicit AlgebraImplementation(context_pointer &&ctx, Args &&...args)
        : storage_base_t(std::forward<Args>(args)...),
          access_layer_t{
              std::move(ctx),
              algebra_info<Impl>::vtype(),
              algebra_info<Impl>::stype(),
              access_layer_t::s_type} {}

    dimn_t size() const override;
    dimn_t dimension() const override;
    bool is_zero() const override;
    optional<deg_t> degree() const override;
    optional<deg_t> width() const override;
    optional<deg_t> depth() const override;

    algebra_t clone() const override;
    algebra_t zero_like() const override;

    scalars::Scalar get(key_type key) const override;
    scalars::Scalar get_mut(key_type key) override;

private:

    std::shared_ptr<AlgebraIteratorInterface>
    make_iterator_ptr(typename Impl::const_iterator it) const;

public:
    AlgebraIterator begin() const override;
    AlgebraIterator end() const override;

    void clear() override;
    void assign(const algebra_interface_t& arg) override;
private:

    template <typename B, typename=traits::enable_if_t<traits::is_detected<dtl::d_has_as_ptr_t, B>::value>>
    optional<scalars::ScalarArray> dense_data_impl(const B& data) const;

    optional<scalars::ScalarArray> dense_data_impl(...) const;

public:

    optional<scalars::ScalarArray> dense_data() const;

private:
    /*
     * The true implementation of operations needs a check of whether the corresponding
     * operation/function is defined for Impl. We're assuming that, at a minimum, Impl
     * has the standard arithmetic operations. But the fused operations need not be
     * defined.
     */

    dtl::ConvertedArgument<Impl> convert_argument(const algebra_interface_t &arg) const;

    void add_scal_mul_impl(const algebra_interface_t& arg, const scalars::Scalar& scalar,
                           dtl::no_implementation);
    void sub_scal_mul_impl(const algebra_interface_t& arg, const scalars::Scalar& scalar,
                           dtl::no_implementation);
    void add_scal_div_impl(const algebra_interface_t& arg, const scalars::Scalar& scalar,
                           dtl::no_implementation);
    void sub_scal_div_impl(const algebra_interface_t& arg, const scalars::Scalar& scalar,
                           dtl::no_implementation);

    void add_mul_impl(const algebra_interface_t& lhs, const algebra_interface_t& rhs,
                      dtl::no_implementation);
    void sub_mul_impl(const algebra_interface_t& lhs, const algebra_interface_t& rhs,
                      dtl::no_implementation);

    void mul_smul_impl(const algebra_interface_t& lhs, const scalars::Scalar& rhs,
                       dtl::no_implementation);
    void mul_sdiv_impl(const algebra_interface_t& lhs, const scalars::Scalar& rhs,
                       dtl::no_implementation);


    void add_scal_mul_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar,
                           dtl::has_implementation);
    void sub_scal_mul_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar,
                           dtl::has_implementation);
    void add_scal_div_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar,
                           dtl::has_implementation);
    void sub_scal_div_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar,
                           dtl::has_implementation);

    void add_mul_impl(const algebra_interface_t &lhs, const algebra_interface_t &rhs,
                      dtl::has_implementation);
    void sub_mul_impl(const algebra_interface_t &lhs, const algebra_interface_t &rhs,
                      dtl::has_implementation);

    void mul_smul_impl(const algebra_interface_t &lhs, const scalars::Scalar &rhs,
                       dtl::has_implementation);
    void mul_sdiv_impl(const algebra_interface_t &lhs, const scalars::Scalar &rhs,
                       dtl::has_implementation);

public:

    algebra_t uminus() const override;
    algebra_t add(const algebra_interface_t& other) const override;
    algebra_t sub(const algebra_interface_t& other) const override;
    algebra_t mul(const algebra_interface_t& other) const override;
    algebra_t smul(const scalars::Scalar& scalar) const override;
    algebra_t sdiv(const scalars::Scalar& scalar) const override;

    void add_inplace(const algebra_interface_t& other) override;
    void sub_inplace(const algebra_interface_t& other) override;
    void mul_inplace(const algebra_interface_t& other) override;
    void smul_inplace(const scalars::Scalar& other) override;
    void sdiv_inplace(const scalars::Scalar& other) override;

    void add_scal_mul(const algebra_interface_t& arg, const scalars::Scalar& scalar) override;
    void sub_scal_mul(const algebra_interface_t& arg, const scalars::Scalar& scalar) override;
    void add_scal_div(const algebra_interface_t& arg, const scalars::Scalar& scalar) override;
    void sub_scal_div(const algebra_interface_t& arg, const scalars::Scalar& scalar) override;

    void add_mul(const algebra_interface_t& lhs, const algebra_interface_t& rhs) override;
    void sub_mul(const algebra_interface_t& lhs, const algebra_interface_t& rhs) override;

    void mul_smul(const algebra_interface_t& lhs, const scalars::Scalar& rhs) override;
    void mul_sdiv(const algebra_interface_t& lhs, const scalars::Scalar& rhs) override;

    std::ostream& print(std::ostream& os) const override;
    bool equals(const algebra_interface_t& other) const override;

};

template <typename Interface, typename Impl, template <typename> class StorageModel>
dimn_t AlgebraImplementation<Interface, Impl, StorageModel>::size() const {
    return data.size();
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
dimn_t AlgebraImplementation<Interface, Impl, StorageModel>::dimension() const {
    return data.dimension();
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
bool AlgebraImplementation<Interface, Impl, StorageModel>::is_zero() const {
    return false;
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
optional<deg_t> AlgebraImplementation<Interface, Impl, StorageModel>::degree() const {
    return algebra_info<Impl>::degree(&data());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
optional<deg_t> AlgebraImplementation<Interface, Impl, StorageModel>::width() const {
    return algebra_info<Impl>::width(&data());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
optional<deg_t> AlgebraImplementation<Interface, Impl, StorageModel>::depth() const {
    return algebra_info<Impl>::max_depth(&data());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
typename Interface::algebra_t AlgebraImplementation<Interface, Impl, StorageModel>::clone() const {
    return algebra_t(Interface::context(), data());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
typename Interface::algebra_t AlgebraImplementation<Interface, Impl, StorageModel>::zero_like() const {
    return algebra_t(Interface::context(), Impl());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
scalars::Scalar AlgebraImplementation<Interface, Impl, StorageModel>::get(key_type key) const {
    using info_t = algebra_info<Impl>;
    auto akey = info_t::convert_key(&data(), key);
    using ref_t = decltype(data()[akey]);
    using trait = scalars::scalar_type_trait<ref_t>;
    return trait::make(data()[akey]);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
scalars::Scalar AlgebraImplementation<Interface, Impl, StorageModel>::get_mut(key_type key) {
    using info_t = algebra_info<Impl>;
    auto akey = info_t::convert_key(&data(), key);
    using ref_t = decltype(data()[akey]);
    using trait = scalars::scalar_type_trait<ref_t>;
    return trait::make(data()[akey]);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
std::shared_ptr<AlgebraIteratorInterface> AlgebraImplementation<Interface, Impl, StorageModel>::make_iterator_ptr(typename Impl::const_iterator it) const {
    return std::shared_ptr<AlgebraIteratorInterface>(
        new AlgebraIteratorImplementation<basis_type, typename Impl::const_iterator>(it, algebra_info<Impl>::basis(data()))
        );
}

template <typename Interface, typename Impl, template <typename> class StorageModel>
AlgebraIterator AlgebraImplementation<Interface, Impl, StorageModel>::begin() const {
    return AlgebraIterator(make_iterator_ptr(data().begin()), bit_cast<std::uintptr_t>(this));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
AlgebraIterator AlgebraImplementation<Interface, Impl, StorageModel>::end() const {
    return AlgebraIterator(make_iterator_ptr(data().end()), bit_cast<std::uintptr_t>(this));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::clear() {
    data().clear();
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::assign(const algebra_interface_t& arg) {
    data() = convert_argument(arg);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
optional<scalars::ScalarArray> AlgebraImplementation<Interface, Impl, StorageModel>::dense_data() const {
    return dense_data_impl(data());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
template <typename B, typename>
optional<scalars::ScalarArray> AlgebraImplementation<Interface, Impl, StorageModel>::dense_data_impl(const B &data) const {
    return {Interface::coeff_type(), data.as_ptr(), data.dimension()};
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
optional<scalars::ScalarArray> AlgebraImplementation<Interface, Impl, StorageModel>::dense_data_impl(...) const {
    return Interface::dense_data();
}

template <typename Interface, typename Impl, template <typename> class StorageModel>
dtl::ConvertedArgument<Impl> AlgebraImplementation<Interface, Impl, StorageModel>::convert_argument(const algebra_interface_t &arg) const {
    if (this->context() == arg.context()) {
        if (this->storage_type() == arg.storage_type()) {
            return algebra_cast<const Impl&>(arg);
        }
        return this->context()->convert(arg, this->storage_type());
    }
    throw std::invalid_argument("cannot convert argument");
}

template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_mul_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar, dtl::no_implementation) {
    Interface::add_scal_mul(arg, scalar);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_mul_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar, dtl::no_implementation) {
    Interface::sub_scal_mul(arg, scalar);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_div_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar, dtl::no_implementation) {
    Interface::add_scal_div(arg, scalar);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_div_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar, dtl::no_implementation) {
    Interface::sub_scal_div(arg, scalar);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_mul_impl(const algebra_interface_t &lhs, const algebra_interface_t &rhs, dtl::no_implementation) {
    Interface::add_mul(lhs, rhs);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_mul_impl(const algebra_interface_t &lhs, const algebra_interface_t &rhs, dtl::no_implementation) {
    Interface::sub_mul(lhs, rhs);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_smul_impl(const algebra_interface_t &lhs, const scalars::Scalar &rhs, dtl::no_implementation) {
    Interface::mul_smul(lhs, rhs);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_sdiv_impl(const algebra_interface_t &lhs, const scalars::Scalar &rhs, dtl::no_implementation) {
    Interface::mul_sdiv(lhs, rhs);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_mul_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar, dtl::has_implementation) {
    data().add_scal_prod(convert_argument(arg), scalars::scalar_cast<scalar_type>(scalar));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_mul_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar, dtl::has_implementation) {
    data().sub_scal_prod(convert_argument(arg, scalars::scalar_cast<scalar_type>(scalar)));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_div_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar, dtl::has_implementation) {
    data().add_scal_div(convert_argument(arg, scalars::scalar_cast<rational_type>(scalar)));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_div_impl(const algebra_interface_t &arg, const scalars::Scalar &scalar, dtl::has_implementation) {
    data().sub_scal_sub(convert_argument(arg, scalars::scalar_cast<rational_type>(scalar)));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_mul_impl(const algebra_interface_t &lhs, const algebra_interface_t &rhs, dtl::has_implementation) {
    data().add_mul(convert_argument(lhs), convert_argument(rhs));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_mul_impl(const algebra_interface_t &lhs, const algebra_interface_t &rhs, dtl::has_implementation) {
    data().sub_mul(convert_argument(lhs), convert_argument(rhs));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_smul_impl(const algebra_interface_t &lhs, const scalars::Scalar &rhs, dtl::has_implementation) {
    data().mul_scal_prod(convert_argument(lhs), scalars::scalar_cast<scalar_type>(rhs));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_sdiv_impl(const algebra_interface_t &lhs, const scalars::Scalar &rhs, dtl::has_implementation) {
    data().mul_scal_div(convert_argument(lhs), scalars::scalar_cast<rational_type>(rhs));
}

template <typename Interface, typename Impl, template <typename> class StorageModel>
typename Interface::algebra_t AlgebraImplementation<Interface, Impl, StorageModel>::uminus() const {
    return algebra_t(Interface::context(), -data());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
typename Interface::algebra_t AlgebraImplementation<Interface, Impl, StorageModel>::add(const algebra_interface_t &other) const {
    return algebra_t(Interface::context(), data() + convert_argument(other));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
typename Interface::algebra_t AlgebraImplementation<Interface, Impl, StorageModel>::sub(const algebra_interface_t &other) const {
    return algebra_t(Interface::context(), data() - convert_argument(other));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
typename Interface::algebra_t AlgebraImplementation<Interface, Impl, StorageModel>::mul(const algebra_interface_t &other) const {
    return algebra_t(Interface::context(), data() * convert_argument(other));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
typename Interface::algebra_t AlgebraImplementation<Interface, Impl, StorageModel>::smul(const scalars::Scalar &scalar) const {
    return algebra_t(Interface::context(), data() * scalars::scalar_cast<scalar_type>(scalar));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
typename Interface::algebra_t AlgebraImplementation<Interface, Impl, StorageModel>::sdiv(const scalars::Scalar &scalar) const {
    return algebra_t(Interface::context(), data() / scalars::scalar_cast<rational_type>(scalar));
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_inplace(const algebra_interface_t &other) {
    data() += convert_argument(other);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_inplace(const algebra_interface_t &other) {
    data() -= convert_argument(other);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_inplace(const algebra_interface_t &other) {
    data() *= convert_argument(other);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::smul_inplace(const scalars::Scalar &other) {
    data() *= scalars::scalar_cast<scalar_type>(other);
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sdiv_inplace(const scalars::Scalar &other) {
    data() /= scalars::scalar_cast<rational_type>(other);
}

template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_mul(const algebra_interface_t &arg, const scalars::Scalar &scalar) {
    add_scal_mul_impl(arg, scalar, dtl::use_impl_t<dtl::d_add_scal_prod, Impl>());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_mul(const algebra_interface_t &arg, const scalars::Scalar &scalar) {
    sub_scal_mul_impl(arg, scalar, dtl::use_impl_t<dtl::d_sub_scal_prod, Impl>());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_scal_div(const algebra_interface_t &arg, const scalars::Scalar &scalar) {
    add_scal_div_impl(arg, scalar, dtl::use_impl_t<dtl::d_add_scal_div, Impl>());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_scal_div(const algebra_interface_t &arg, const scalars::Scalar &scalar) {
    sub_scal_div_impl(arg, scalar, dtl::use_impl_t<dtl::d_sub_scal_div, Impl>());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::add_mul(const algebra_interface_t &lhs, const algebra_interface_t &rhs) {
    add_mul_impl(lhs, rhs, dtl::use_impl_t<dtl::d_add_mul, Impl>());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::sub_mul(const algebra_interface_t &lhs, const algebra_interface_t &rhs) {
    sub_mul_impl(lhs, rhs, dtl::use_impl_t<dtl::d_sub_mul, Impl>());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_smul(const algebra_interface_t &lhs, const scalars::Scalar &rhs) {
    mul_smul_impl(lhs, rhs, dtl::use_impl_t<dtl::d_mul_scal_prod, Impl>());
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
void AlgebraImplementation<Interface, Impl, StorageModel>::mul_sdiv(const algebra_interface_t &lhs, const scalars::Scalar &rhs) {
    mul_sdiv_impl(lhs, rhs, dtl::use_impl_t<dtl::d_mul_scal_div, Impl>());
}

template <typename Interface, typename Impl, template <typename> class StorageModel>
std::ostream &AlgebraImplementation<Interface, Impl, StorageModel>::print(std::ostream &os) const {
    return os << data();
}
template <typename Interface, typename Impl, template <typename> class StorageModel>
bool AlgebraImplementation<Interface, Impl, StorageModel>::equals(const algebra_interface_t &other) const {
    return data() == convert_argument(other);
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_IMPL_H_
