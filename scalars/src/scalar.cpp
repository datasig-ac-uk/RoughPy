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
// Created by user on 26/02/23.
//

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_pointer.h>
#include <roughpy/scalars/scalar_type.h>

#include <ostream>
#include <roughpy/platform/serialization.h>

using namespace rpy;
using namespace rpy::scalars;

Scalar::Scalar(ScalarPointer other, uint32_t new_flags) : ScalarPointer(other)
{
    m_flags = new_flags;
}
Scalar::Scalar(ScalarPointer data, flags::PointerType ptype)
    : ScalarPointer(data)
{
    m_flags |= ptype;
}

Scalar::Scalar(ScalarInterface* other)
{
    if (other == nullptr) {
        RPY_THROW(std::invalid_argument, "scalar interface pointer cannot be null");
    }
    p_type = other->type();
    p_data = other;
    m_flags |= flags::InterfacePointer
            | (other->is_const() ? flags::IsConst : flags::IsMutable);
}
Scalar::Scalar(ScalarPointer ptr) : ScalarPointer(ptr)
{
    if (p_data != nullptr && p_type == nullptr) {
        RPY_THROW(std::runtime_error, "non-zero scalars must have a type");
    }
}

Scalar::Scalar(const ScalarType* type)
    : ScalarPointer(type, static_cast<void*>(nullptr))
{}
Scalar::Scalar(scalar_t scal)
    : ScalarPointer(ScalarType::of<scalar_t>()->allocate(1))
{
    p_type->convert_copy(to_mut_pointer(), {p_type, &scal}, 1);
}
Scalar::Scalar(const ScalarType* type, scalar_t scal)
    : ScalarPointer(type->allocate(1))
{
    const auto* scal_type = ScalarType::of<scalar_t>();
    p_type->convert_copy(const_cast<void*>(p_data), {scal_type, &scal}, 1);
}
Scalar::Scalar(const Scalar& other)
    : ScalarPointer(other.p_type == nullptr ? ScalarPointer()
                                            : other.p_type->allocate(1))
{
    if (p_type != nullptr) {
        p_type->convert_copy(to_mut_pointer(), other.to_pointer(), 1);
    }
}
Scalar::Scalar(Scalar&& other) noexcept : ScalarPointer(std::move(other))
{
    /*
     * Since other might own its pointer, we need to make sure
     * the pointer is set to null before the destructor on other
     * is called.
     */
    other.p_data = nullptr;
}

Scalar::~Scalar()
{
    if (p_data != nullptr) {
        if (is_interface()) {
            delete static_cast<ScalarInterface*>(const_cast<void*>(p_data));
        } else if (is_owning()) {
            p_type->free(to_mut_pointer(), 1);
        }
        p_data = nullptr;
    }
}

bool Scalar::is_value() const noexcept
{
    if (p_data == nullptr) { return true; }
    if (is_interface()) {
        return static_cast<const ScalarInterface*>(p_data)->is_value();
    }

    return is_owning();
}
bool Scalar::is_zero() const noexcept
{
    if (p_data == nullptr) { return true; }
    if (is_interface()) {
        return static_cast<const ScalarInterface*>(p_data)->is_zero();
    }

    // TODO: finish this off?
    return p_type->is_zero(to_pointer());
}

Scalar& Scalar::operator=(const Scalar& other)
{
    if (is_const()) {
        RPY_THROW(std::runtime_error, "Cannot cast to a const value");
    }
    if (this != std::addressof(other)) {
        if (is_interface()) {
            auto* iface
                    = static_cast<ScalarInterface*>(const_cast<void*>(p_data));
            iface->assign(other.to_pointer());
        } else {
            p_type->convert_copy(to_mut_pointer(), other.to_pointer(), 1);
        }
    }
    return *this;
}
Scalar& Scalar::operator=(Scalar&& other) noexcept
{
    if (this != std::addressof(other)) {
        if (p_type == nullptr || is_const()) {
            this->~Scalar();
            p_data = other.p_data;
            p_type = other.p_type;
            m_flags = other.m_flags;
            other.p_data = nullptr;
            other.p_type = nullptr;
        } else {
            if (is_interface()) {
                auto* iface = static_cast<ScalarInterface*>(
                        const_cast<void*>(p_data));
                iface->assign(other.to_pointer());
            } else {
                p_type->convert_copy(to_mut_pointer(), other.to_pointer(), 1);
            }
        }
    }

    return *this;
}

ScalarPointer Scalar::to_pointer() const noexcept
{
    if (is_interface()) {
        return static_cast<const ScalarInterface*>(p_data)->to_pointer();
    }
    return {p_type, p_data};
}
ScalarPointer Scalar::to_mut_pointer()
{
    if (is_const()) {
        RPY_THROW(std::runtime_error, "cannot get non-const pointer to const value");
    }
    auto* ptr = const_cast<void*>(p_data);
    if (is_interface()) {
        return static_cast<ScalarInterface*>(ptr)->to_pointer();
    }
    return {p_type, ptr};
}
void Scalar::set_to_zero()
{
    if (p_data == nullptr) {
        RPY_CHECK(p_type != nullptr);
        RPY_CHECK(!is_const());
        RPY_CHECK(is_owning());
        ScalarPointer::operator=(p_type->allocate(1));
        p_type->assign(to_mut_pointer(), 0, 1);
    }
    // TODO: look at the logic here.
}
scalar_t Scalar::to_scalar_t() const
{
    if (p_data == nullptr) { return scalar_t(0); }
    if (is_interface()) {
        return static_cast<const ScalarInterface*>(p_data)->as_scalar();
    }
    RPY_CHECK(p_type != nullptr);
    return p_type->to_scalar_t(to_pointer());
}
Scalar Scalar::operator-() const
{
    if (p_data == nullptr) { return Scalar(p_type); }
    if (is_interface()) {
        return static_cast<const ScalarInterface*>(p_data)->uminus();
    }
    return p_type->uminus(to_pointer());
}

#define RPY_SCALAR_OP(OP, MNAME)                                               \
    Scalar Scalar::operator OP(const Scalar& other) const                      \
    {                                                                          \
        const ScalarType* type = (p_type != nullptr) ? p_type : other.p_type;  \
        if (type == nullptr) { return Scalar(); }                              \
        return type->MNAME(to_pointer(), other.to_pointer());                  \
    }

RPY_SCALAR_OP(+, add)
RPY_SCALAR_OP(-, sub)
RPY_SCALAR_OP(*, mul)

#undef RPY_SCALAR_OP

Scalar Scalar::operator/(const Scalar& other) const
{
    const ScalarType* type = (p_type != nullptr) ? p_type : other.p_type;
    if (type == nullptr) { return Scalar(); }
    if (other.p_data == nullptr) {
        RPY_THROW(std::runtime_error, "division by zero");
    }

    return type->div(to_pointer(), other.to_pointer());
}

#define RPY_SCALAR_IOP(OP, MNAME)                                              \
    Scalar& Scalar::operator OP(const Scalar& other)                           \
    {                                                                          \
        if (is_const()) {                                                      \
            RPY_THROW(std::runtime_error,                                          \
                    "performing inplace operation on const scalar");           \
        }                                                                      \
                                                                               \
        if (p_type == nullptr) {                                               \
            RPY_DBG_ASSERT(p_data == nullptr);                                 \
            /* We just established that other.p_data != nullptr */             \
            RPY_DBG_ASSERT(other.p_type != nullptr);                           \
            p_type = other.p_type;                                             \
        }                                                                      \
        if (p_data == nullptr) {                                               \
            if (p_type == nullptr) { p_type = other.p_type; }                  \
            set_to_zero();                                                     \
        }                                                                      \
        if (is_interface()) {                                                  \
            auto* iface = static_cast<ScalarInterface*>(                       \
                    const_cast<void*>(p_data));                                \
            iface->MNAME##_inplace(other);                                     \
        } else {                                                               \
            p_type->MNAME##_inplace(to_mut_pointer(), other.to_pointer());     \
        }                                                                      \
        return *this;                                                          \
    }

RPY_SCALAR_IOP(+=, add)
RPY_SCALAR_IOP(-=, sub)
RPY_SCALAR_IOP(*=, mul)

Scalar& Scalar::operator/=(const Scalar& other)
{
    if (is_const()) {
        RPY_THROW(std::runtime_error,
                "performing inplace operation on const scalar");
    }
    if (other.p_data == nullptr) {
        RPY_THROW(std::runtime_error, "division by zero");
    }
    if (p_type == nullptr) {
        RPY_DBG_ASSERT(p_data == nullptr);
        RPY_DBG_ASSERT(other.p_type != nullptr);
        p_type = other.p_type;
    }
    if (p_data == nullptr) {
        if (p_type == nullptr) { p_type = other.p_type->rational_type(); }
        set_to_zero();
    }
    if (is_interface()) {
        auto* iface = static_cast<ScalarInterface*>(const_cast<void*>(p_data));
        iface->div_inplace(other);
    } else {
        p_type->rational_type()->div_inplace(to_mut_pointer(),
                                             other.to_pointer());
    }
    return *this;
}

#undef RPY_SCALAR_IOP

bool Scalar::operator==(const Scalar& rhs) const noexcept
{
    if (p_type == nullptr) { return rhs.is_zero(); }
    return p_type->are_equal(to_pointer(), rhs.to_pointer());
}
bool Scalar::operator!=(const Scalar& rhs) const noexcept
{
    return !operator==(rhs);
}
std::ostream& rpy::scalars::operator<<(std::ostream& os, const Scalar& arg)
{
    if (arg.type() == nullptr) {
        os << '0';
    } else {
        arg.type()->print(arg.to_pointer(), os);
    }

    return os;
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::Scalar
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>
