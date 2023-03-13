//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_SPARSE_MUTABLE_REF_SCALAR_TRAIT_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_SPARSE_MUTABLE_REF_SCALAR_TRAIT_H

#include <roughpy/scalars/scalar_interface.h>
#include <roughpy/scalars/scalar_traits.h>

#include <libalgebra_lite/sparse_vector.h>

namespace rpy {
namespace scalars {
namespace dtl {

template <typename MapType, typename KeyType>
class SparseMutableRefScalarImpl : public ScalarInterface {
    using data_type = lal::dtl::sparse_mutable_reference<MapType, KeyType>;
    using trait = scalar_type_trait<data_type>;

    data_type m_data;

public:

    explicit SparseMutableRefScalarImpl(data_type&& arg) : m_data(std::move(arg))
    {}

    using value_type = typename MapType::mapped_type;
    using rational_type = typename trait::rational_type;

    const ScalarType *type() const noexcept override {
        return ScalarType::of<value_type>();
    }
    bool is_const() const noexcept override {
        return false;
    }
    bool is_value() const noexcept override {
        return false;
    }
    bool is_zero() const noexcept override {
        return static_cast<const value_type&>(m_data) == value_type(0);
    }
    scalar_t as_scalar() const override {
        return scalar_t(static_cast<const value_type&>(m_data));
    }
    void assign(ScalarPointer pointer) override {
        value_type tmp = static_cast<const value_type&>(m_data);
        type()->convert_copy({type(), &tmp}, pointer, 1);
        m_data = tmp;
    }
    void assign(const Scalar &other) override {
        assign(other.to_pointer());
    }
    void assign(const void *data, const std::string &type_id) override {
        value_type tmp = static_cast<const value_type &>(m_data);
        type()->convert_copy({type(), &tmp}, data, 1, type_id);
        m_data = tmp;
    }
    ScalarPointer to_pointer() override {
        throw std::runtime_error("cannot get non-const pointer to proxy reference type");
    }
    ScalarPointer to_pointer() const noexcept override {
        return {type(), &static_cast<const value_type &>(m_data)};
    }
private:

    template <typename F>
    void inplace_function(const Scalar& other, F&& f) {
        value_type tmp(0);
        type()->convert_copy({type(), &tmp}, other.to_pointer(), 1);
        m_data = f(static_cast<const value_type&>(m_data), tmp);
    }

public:
    void add_inplace(const Scalar &other) override {
        inplace_function(other, std::plus<value_type>());
    }
    void sub_inplace(const Scalar &other) override {
        inplace_function(other, std::minus<value_type>());
    }
    void mul_inplace(const Scalar &other) override {
        inplace_function(other, std::multiplies<value_type>());
    }
    void div_inplace(const Scalar &other) override {
        rational_type tmp(1);
        type()->rational_type()->convert_copy({type()->rational_type(), &tmp}, other.to_pointer(), 1);
        m_data /= tmp;
    }

    std::ostream &print(std::ostream &os) const override {
        return os << static_cast<const value_type &>(m_data);
    }
};

}// namespace dtl

template <typename MapType, typename KeyType>
class scalar_type_trait<lal::dtl::sparse_mutable_reference<MapType, KeyType>> {
public:
    using value_type = typename MapType::mapped_type;
    using rational_type = typename lal::coefficient_trait<value_type>::rational_type;
    using reference = lal::dtl::sparse_mutable_reference<MapType, KeyType>;

    static const ScalarType* get_type() noexcept {
        return ScalarType::of<value_type>();
    }

    static Scalar make(reference arg) {
        return Scalar(new dtl::SparseMutableRefScalarImpl<MapType, KeyType>(std::move(arg)));
    }

};


}// namespace scalars
}// namespace rpy

#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_SPARSE_MUTABLE_REF_SCALAR_TRAIT_H
