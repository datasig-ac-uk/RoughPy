// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 28/04/23.
//

#ifndef ROUGHPY_DEVICE_DEVICE_ALGEBRA_BASE_H
#define ROUGHPY_DEVICE_DEVICE_ALGEBRA_BASE_H

#include <roughpy/algebra/algebra_base.h>
#include <roughpy/algebra/algebra_impl.h>
#include <roughpy/algebra/algebra_iterator.h>
#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

#include "device_context.h"

namespace rpy {
namespace device {

namespace dtl {
class PossiblyConverted : public scalars::ScalarArray
{
public:
    using scalars::ScalarArray::ScalarArray;
    using scalars::ScalarArray::operator=;

    ~PossiblyConverted();
};

}// namespace dtl

template <typename Interface, typename Derived>
class DeviceAlgebraBase : public Interface
{
protected:
    using algebra_t = typename Interface::algebra_t;
    using basis_t = typename Interface::basis_t;
    using key_type = typename basis_t::key_type;

    scalars::ScalarArray m_data;
    const DeviceContext* p_dctx;

    dtl::PossiblyConverted try_convert(const algebra_t& arg) const;

    dtl::PossiblyConverted borrow_on_host() const;

    void resize(dimn_t new_size);

public:
    using const_iterator = algebra::AlgebraIterator<algebra_t>;

    DeviceAlgebraBase(scalars::ScalarArray&& data, const DeviceContext* ctx)
        : Interface(
                ctx->context(), algebra::VectorType::Dense, ctx->ctype(),
                data.is_owning() ? algebra::ImplementationType::DeviceOwned
                                 : algebra::ImplementationType::DeviceBorrowed
        ),
          m_data(std::move(data)), p_dctx(ctx)
    {}

    dimn_t dimension() const override;
    dimn_t size() const override;
    bool is_zero() const override;

    algebra_t clone() const override;
    algebra_t zero_like() const override;

    algebra_t borrow() const override;
    algebra_t borrow_mut() override;

    void clear() override;
    void assign(const algebra_t& other) override;

    std::ostream& print(std::ostream& os) const override;

    bool equals(const algebra_t& other) const override;

    scalars::Scalar get(key_type key) const override;
    scalars::Scalar get_mut(key_type key) override;

    const_iterator begin() const override;
    const_iterator end() const override;

    optional<scalars::ScalarArray> dense_data() const override;

    algebra_t uminus() const override;
    algebra_t add(const algebra_t& other) const override;
    algebra_t sub(const algebra_t& other) const override;
    //    algebra_t mul(const algebra_t& other) const override;
    algebra_t smul(const scalars::Scalar& other) const override;
    algebra_t sdiv(const scalars::Scalar& other) const override;

    void add_inplace(const algebra_t& other) override;
    void sub_inplace(const algebra_t& other) override;
    //    void mul_inplace(const algebra_t& other) override;
    void smul_inplace(const scalars::Scalar& other) override;
    void sdiv_inplace(const scalars::Scalar& other) override;

    void
    add_scal_mul(const algebra_t& rhs, const scalars::Scalar& scalar) override;
    void
    sub_scal_mul(const algebra_t& rhs, const scalars::Scalar& scalar) override;
    void
    add_scal_div(const algebra_t& rhs, const scalars::Scalar& scalar) override;
    void
    sub_scal_div(const algebra_t& rhs, const scalars::Scalar& scalar) override;

    //    void add_mul(const algebra_t& lhs, const algebra_t& rhs) override;
    //    void sub_mul(const algebra_t& lhs, const algebra_t& rhs) override;
    //    void mul_smul(const algebra_t& lhs, const scalars::Scalar& rhs)
    //    override; void mul_sdiv(const algebra_t& lhs, const scalars::Scalar&
    //    rhs) override;
    //
};

template <typename Interface, typename Derived>
dtl::PossiblyConverted
DeviceAlgebraBase<Interface, Derived>::try_convert(const algebra_t& arg) const
{
    const auto* type = p_dctx->ctype();

    auto dense_data_op = arg.dense_data();
    RPY_CHECK(dense_data_op.has_value());
    const auto dense_data = *dense_data_op;

    dtl::PossiblyConverted result(type);
    const auto size = dense_data.size();
    if (size == 0) { return result; }

    if (dense_data.type() == type) {
        result = dense_data.borrow();
        return result;
    }

    result = scalars::ScalarArray(type->allocate(size), size);
    type->convert_copy(scalars::ScalarPointer(result), dense_data, size);
    return result;
}
template <typename Interface, typename Derived>
dtl::PossiblyConverted
DeviceAlgebraBase<Interface, Derived>::borrow_on_host() const
{
    if (p_dctx != nullptr && m_data) {
        const auto* type = m_data.type();
        const auto* host_type = type->host_type();
        const auto size = m_data.size();

        dtl::PossiblyConverted result(host_type->allocate(size), size);
        type->convert_copy(
                scalars::ScalarPointer(result), scalars::ScalarPointer(m_data),
                size
        );
        return result;
    }
    return {};
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::resize(dimn_t new_size)
{
    RPY_CHECK(new_size >= m_data.size());
    const auto* type = p_dctx->ctype();

    scalars::ScalarArray array(type->allocate(new_size), new_size);
    type->convert_copy(array, m_data, m_data.size());
    m_data = std::move(array);
}
template <typename Interface, typename Derived>
dimn_t DeviceAlgebraBase<Interface, Derived>::dimension() const
{
    return m_data.size();
}
template <typename Interface, typename Derived>
dimn_t DeviceAlgebraBase<Interface, Derived>::size() const
{
    return p_dctx->count_nonzero(m_data);
}
template <typename Interface, typename Derived>
bool DeviceAlgebraBase<Interface, Derived>::is_zero() const
{
    if (m_data.size() > 0) { return p_dctx->is_zero(m_data); }
    return false;
}

template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::clone() const
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    if (m_data) {
        const auto* type = m_data.type();
        scalars::ScalarArray new_data(
                type->allocate(m_data.size()), m_data.size()
        );
        p_dctx->unary_op(new_data, m_data, DeviceContext::Clone, {});
        return algebra_t(new Derived(std::move(new_data), p_dctx));
    }
    return algebra_t(Interface::context());
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::zero_like() const
{
    return algebra_t(Interface::context());
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::borrow() const
{
    return algebra_t(new Derived(m_data.borrow(), p_dctx));
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::borrow_mut()
{
    return algebra_t(new Derived(m_data.borrow_mut(), p_dctx));
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::clear()
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());
    if (m_data) { p_dctx->ctype()->assign(m_data, 0, 1); }
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::assign(const algebra_t& other)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());
    const auto* type = p_dctx->ctype();

    const auto other_data = try_convert(other);
    const auto size = other_data.size();
    if (size > m_data.size()) {
        resize(size);
    } else if (size < m_data.size()) {
        if (m_data.is_owning()) { type->free(m_data, m_data.size()); }
        m_data = {type->allocate(size), size};
    }

    p_dctx->unary_op(m_data, other_data, DeviceContext::Clone, {});
}
template <typename Interface, typename Derived>
std::ostream& DeviceAlgebraBase<Interface, Derived>::print(std::ostream& os
) const
{
    os << "{ ";
    if (m_data) {
        const auto borrowed = borrow_on_host();
        const auto* type = borrowed.type();
        const auto& basis = Interface::basis();

        scalars::ScalarPointer ptr(borrowed);
        for (dimn_t idx = 0; idx < borrowed.size(); ++idx, ++ptr) {
            type->print(ptr, os);
            os << '(' << basis.key_to_string(basis.index_to_key(idx)) << ") ";
        }
    }
    os << '}';
    return os;
}
template <typename Interface, typename Derived>
bool DeviceAlgebraBase<Interface, Derived>::equals(const algebra_t& other) const
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    const auto other_data = try_convert(other);
    return p_dctx->equals(m_data, other_data);
}
template <typename Interface, typename Derived>
scalars::Scalar DeviceAlgebraBase<Interface, Derived>::get(key_type key) const
{
    return m_data[Interface::basis().key_to_index(key)];
}
template <typename Interface, typename Derived>
scalars::Scalar DeviceAlgebraBase<Interface, Derived>::get_mut(key_type key)
{
    RPY_CHECK(!m_data.is_const());
    return m_data[Interface::basis().key_to_index(key)];
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::const_iterator
DeviceAlgebraBase<Interface, Derived>::begin() const
{
    RPY_UNREACHABLE();
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::const_iterator
DeviceAlgebraBase<Interface, Derived>::end() const
{
    RPY_UNREACHABLE();
}
template <typename Interface, typename Derived>
optional<scalars::ScalarArray>
DeviceAlgebraBase<Interface, Derived>::dense_data() const
{
    return m_data.borrow();
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::uminus() const
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    if (m_data) {
        const auto* type = m_data.type();
        scalars::ScalarArray new_data(
                type->allocate(m_data.size()), m_data.size()
        );
        p_dctx->unary_op(new_data, m_data, DeviceContext::UMinus, {});
        return algebra_t(new Derived(std::move(new_data), p_dctx));
    }
    return algebra_t(Interface::context());
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::add(const algebra_t& other) const
{
    // TODO: Clean this up
    RPY_DBG_ASSERT(p_dctx != nullptr);
    const auto* type = p_dctx->ctype();

    if (!m_data || m_data.size() == 0) {
        auto other_dd = other.dense_data();
        if (!other_dd) { return algebra_t(); }
        auto dense_data = *other_dd;
        const auto size = dense_data.size();

        scalars::ScalarArray array(type->allocate(size), size);
        type->convert_copy(array, dense_data, size);
        return algebra_t(new Derived(std::move(array), p_dctx));
    }

    if (other.dimension() == 0) { return clone(); }

    RPY_CHECK(m_data.size() == other.dimension());
    auto other_data = try_convert(other);
    const auto size = m_data.size();

    scalars::ScalarArray array(type->allocate(size), size);
    p_dctx->binary_op(array, m_data, other_data, DeviceContext::Add, {});

    return algebra_t(new Derived(std::move(array), p_dctx));
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::sub(const algebra_t& other) const
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    const auto* type = p_dctx->ctype();

    if (other.dimension() == 0) { return clone(); }
    const auto other_data = try_convert(other);
    const auto size = other_data.size();

    if (!m_data) {
        scalars::ScalarArray array(type->allocate(size), size);
        p_dctx->unary_op(array, other_data, DeviceContext::UMinus, {});
        return algebra_t(new Derived(std::move(array), p_dctx));
    }

    RPY_CHECK(m_data.size() == size);
    scalars::ScalarArray array(type->allocate(size), size);
    p_dctx->binary_op(array, m_data, other_data, DeviceContext::Sub, {});

    return algebra_t(new Derived(std::move(array), p_dctx));
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::smul(const scalars::Scalar& other) const
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    if (m_data) {
        const auto* type = m_data.type();
        scalars::ScalarArray new_data(
                type->allocate(m_data.size()), m_data.size()
        );
        p_dctx->unary_op(new_data, m_data, DeviceContext::SMul, other);
        return algebra_t(new Derived(std::move(new_data), p_dctx));
    }
    return algebra_t(Interface::context());
}
template <typename Interface, typename Derived>
typename DeviceAlgebraBase<Interface, Derived>::algebra_t
DeviceAlgebraBase<Interface, Derived>::sdiv(const scalars::Scalar& other) const
{
    if (p_dctx != nullptr && m_data) {
        const auto* type = m_data.type();
        scalars::ScalarArray new_data(
                type->allocate(m_data.size()), m_data.size()
        );
        p_dctx->unary_op(new_data, m_data, DeviceContext::SDiv, other);
        return algebra_t(new Derived(std::move(new_data), p_dctx));
    }
    return algebra_t(Interface::context());
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::add_inplace(const algebra_t& other)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());

    const auto other_data = try_convert(other);
    const auto size = other_data.size();

    if (m_data.size() < size) { resize(size); }

    p_dctx->inplace_binary_op(
            m_data, other_data, DeviceContext::InplaceAdd, {}, {}
    );
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::sub_inplace(const algebra_t& other)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());

    const auto other_data = try_convert(other);
    const auto size = other_data.size();

    if (m_data.size() < size) { resize(size); }

    p_dctx->inplace_binary_op(
            m_data, other_data, DeviceContext::InplaceSub, {}, {}
    );
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::smul_inplace(
        const scalars::Scalar& other
)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());

    if (m_data) {
        p_dctx->inplace_unary_op(m_data, DeviceContext::InplaceSMul, other);
    }
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::sdiv_inplace(
        const scalars::Scalar& other
)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    if (m_data) {
        p_dctx->inplace_unary_op(m_data, DeviceContext::InplaceSDiv, other);
    }
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::add_scal_mul(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());

    const auto other_data = try_convert(rhs);
    const auto size = other_data.size();

    if (m_data.size() < size) { resize(size); }

    p_dctx->inplace_binary_op(
            m_data, other_data, DeviceContext::InplaceAddSMul, {}, scalar
    );
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::sub_scal_mul(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());

    const auto other_data = try_convert(rhs);
    const auto size = other_data.size();

    if (m_data.size() < size) { resize(size); }

    p_dctx->inplace_binary_op(
            m_data, other_data, DeviceContext::InplaceSubSMul, {}, scalar
    );
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::add_scal_div(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());

    const auto other_data = try_convert(rhs);
    const auto size = other_data.size();

    if (m_data.size() < size) { resize(size); }

    p_dctx->inplace_binary_op(
            m_data, other_data, DeviceContext::InplaceAddSDiv, {}, scalar
    );
}
template <typename Interface, typename Derived>
void DeviceAlgebraBase<Interface, Derived>::sub_scal_div(
        const algebra_t& rhs, const scalars::Scalar& scalar
)
{
    RPY_DBG_ASSERT(p_dctx != nullptr);
    RPY_CHECK(!m_data.is_const());

    const auto other_data = try_convert(rhs);
    const auto size = other_data.size();

    if (m_data.size() < size) { resize(size); }

    p_dctx->inplace_binary_op(
            m_data, other_data, DeviceContext::InplaceSubSDiv, {}, scalar
    );
}

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_DEVICE_ALGEBRA_BASE_H
