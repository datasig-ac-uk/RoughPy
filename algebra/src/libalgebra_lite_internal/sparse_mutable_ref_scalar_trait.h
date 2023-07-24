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

template <typename Vector>
class SparseMutableRefScalarImpl : public ScalarInterface
{
    using data_type = lal::dtl::sparse_mutable_reference<Vector>;
    using trait = scalar_type_trait<data_type>;

    data_type m_data;

public:
    explicit SparseMutableRefScalarImpl(data_type&& arg)
        : m_data(std::move(arg))
    {}

    using value_type = typename data_type::scalar_type;
    using rational_type = typename trait::rational_type;

    const ScalarType* type() const noexcept override
    {
        return ScalarType::of<value_type>();
    }
    bool is_const() const noexcept override { return false; }
    bool is_value() const noexcept override { return false; }
    bool is_zero() const noexcept override
    {
        return static_cast<const value_type&>(m_data) == value_type(0);
    }

private:
    template <typename T>
    static enable_if_t<is_convertible<const T&, scalar_t>::value, scalar_t>
    convert_to_scalar_t(const T& arg)
    {
        return static_cast<scalar_t>(arg);
    }

    template <typename T>
    static enable_if_t<!is_convertible<const T&, scalar_t>::value, scalar_t>
    convert_to_scalar_t(const T&)
    {
        RPY_THROW(std::runtime_error, "cannot convert to scalar_t");
    }

public:
    scalar_t as_scalar() const override
    {
        return convert_to_scalar_t(static_cast<const value_type&>(m_data));
    }
    void assign(ScalarPointer pointer) override
    {
        value_type tmp = static_cast<const value_type&>(m_data);
        type()->convert_copy({type(), &tmp}, pointer, 1);
        m_data = tmp;
    }
    void assign(const Scalar& other) override { assign(other.to_pointer()); }
    void assign(const void* data, const string& type_id) override
    {
        value_type tmp = static_cast<const value_type&>(m_data);
        type()->convert_copy({type(), &tmp}, data, 1, type_id);
        m_data = tmp;
    }
    ScalarPointer to_pointer() override
    {
        RPY_THROW(std::runtime_error,
                "cannot get non-const pointer to proxy reference type"
        );
    }
    ScalarPointer to_pointer() const noexcept override
    {
        return {type(), &static_cast<const value_type&>(m_data)};
    }

private:
    template <typename F>
    void inplace_function(const Scalar& other, F&& f)
    {
        value_type tmp(0);
        type()->convert_copy({type(), &tmp}, other.to_pointer(), 1);
        m_data = f(static_cast<const value_type&>(m_data), tmp);
    }

public:
    void add_inplace(const Scalar& other) override
    {
        inplace_function(other, std::plus<value_type>());
    }
    void sub_inplace(const Scalar& other) override
    {
        inplace_function(other, std::minus<value_type>());
    }
    void mul_inplace(const Scalar& other) override
    {
        inplace_function(other, std::multiplies<value_type>());
    }
    void div_inplace(const Scalar& other) override
    {
        rational_type tmp(1);
        type()->rational_type()->convert_copy(
                {type()->rational_type(), &tmp}, other.to_pointer(), 1
        );
        m_data /= tmp;
    }

    std::ostream& print(std::ostream& os) const override
    {
        return os << static_cast<const value_type&>(m_data);
    }
};

}// namespace dtl

template <typename Vector>
class scalar_type_trait<lal::dtl::sparse_mutable_reference<Vector>>
{
    using mutable_ref_type = lal::dtl::sparse_mutable_reference<Vector>;

public:
    using value_type = typename mutable_ref_type::scalar_type;
    using rational_type =
            typename lal::coefficient_trait<value_type>::rational_type;
    using reference = lal::dtl::sparse_mutable_reference<Vector>;

    static const ScalarType* get_type() noexcept
    {
        return ScalarType::of<value_type>();
    }

    static Scalar make(reference arg)
    {
        return Scalar(new dtl::SparseMutableRefScalarImpl<Vector>(std::move(arg)
        ));
    }
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_SPARSE_MUTABLE_REF_SCALAR_TRAIT_H
