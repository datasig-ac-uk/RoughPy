//
// Created by user on 26/02/23.
//

#include "scalar.h"

#include <ostream>

using namespace rpy::scalars;

Scalar::Scalar(ScalarPointer data, PointerType ptype)
    : ScalarPointer(data), m_pointer_type(ptype) {
}

Scalar::Scalar(ScalarInterface *other)
      : m_pointer_type(InterfacePointer)
{
    if (other == nullptr) {
        throw std::invalid_argument("scalar interface pointer cannot be null");
    }
    p_type = other->type();
    p_data = other;
    m_constness = other->is_const() ? IsConst : IsMutable;
}
Scalar::Scalar(ScalarPointer ptr)
    : ScalarPointer(ptr),
      m_pointer_type(BorrowedPointer) {
    if (p_data != nullptr && p_type == nullptr) {
        throw std::runtime_error("non-zero scalars must have a type");
    }
}

Scalar::Scalar(const ScalarType *type)
    : ScalarPointer(type, nullptr, IsMutable),
      m_pointer_type(OwnedPointer) {
}
Scalar::Scalar(scalar_t scal)
    : ScalarPointer(ScalarType::of<scalar_t>()->allocate(1)),
      m_pointer_type(OwnedPointer) {
    p_type->convert_copy(to_mut_pointer(), {p_type, &scal}, 1);
}
Scalar::Scalar(const ScalarType *type, scalar_t scal)
    : ScalarPointer(type->allocate(1)),
      m_pointer_type(OwnedPointer) {
    const auto *scal_type = ScalarType::of<scalar_t>();
    p_type->convert_copy(const_cast<void *>(p_data), {scal_type, &scal}, 1);
}
Scalar::Scalar(const Scalar &other)
    : ScalarPointer(other.p_type != nullptr ? ScalarPointer() : other.p_type->allocate(1)),
      m_pointer_type(other.m_pointer_type) {
    if (p_type != nullptr) {
        p_type->convert_copy(to_mut_pointer(), other.to_pointer(), 1);
    }
}
Scalar::Scalar(Scalar &&other) noexcept
    : ScalarPointer(other),
      m_pointer_type(other.m_pointer_type) {
    /*
     * Since other might own its pointer, we need to make sure
     * the pointer is set to null before the destructor on other
     * is called.
     */
    other.p_data = nullptr;
}

Scalar::~Scalar() {
    if (p_data != nullptr) {
        if (m_pointer_type == InterfacePointer) {
            delete static_cast<ScalarInterface *>(const_cast<void *>(p_data));
        } else if (m_pointer_type == OwnedPointer) {
            p_type->free(to_mut_pointer(), 1);
        }
        p_data = nullptr;
    }
}

bool Scalar::is_value() const noexcept {
    if (p_data == nullptr) {
        return true;
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const ScalarInterface *>(p_data)->is_value();
    }

    return m_pointer_type == OwnedPointer;
}
bool Scalar::is_zero() const noexcept {
    if (p_data == nullptr) {
        return true;
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const ScalarInterface *>(p_data)->is_zero();
    }
    if (m_pointer_type == OwnedPointer) {
        return true;
    }

    // TODO: finish this off?
    return p_type->is_zero(to_pointer());
}

Scalar &Scalar::operator=(const Scalar &other) {
    if (m_constness == IsConst) {
        throw std::runtime_error("Cannot cast to a const value");
    }
    if (this != std::addressof(other)) {
        if (m_pointer_type == InterfacePointer) {
            auto *iface = static_cast<ScalarInterface *>(const_cast<void *>(p_data));
            iface->assign(other.to_pointer());
        } else {
            p_type->convert_copy(to_mut_pointer(), other.to_pointer(), 1);
        }
    }
    return *this;
}
Scalar &Scalar::operator=(Scalar &&other) noexcept {
    if (this != std::addressof(other)) {
        if (p_type == nullptr || m_constness == IsConst) {
            this->~Scalar();
            p_data = other.p_data;
            p_type = other.p_type;
            m_constness = other.m_constness;
            m_pointer_type = other.m_pointer_type;
            other.p_data = nullptr;
            other.p_type = nullptr;
            other.m_constness = IsConst;
            other.m_pointer_type = BorrowedPointer;
        } else {
            if (m_pointer_type == InterfacePointer) {
                auto *iface = static_cast<ScalarInterface *>(const_cast<void *>(p_data));
                iface->assign(other.to_pointer());
            } else {
                p_type->convert_copy(to_mut_pointer(), other.to_pointer(), 1);
            }
        }
    }

    return *this;
}

ScalarPointer Scalar::to_pointer() const noexcept {
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const ScalarInterface *>(p_data)->to_pointer();
    }
    return {p_type, p_data};
}
ScalarPointer Scalar::to_mut_pointer(){
    if (m_constness == IsConst) {
        throw std::runtime_error("cannot get non-const pointer to const value");
    }
    auto* ptr = const_cast<void*>(p_data);
    if (m_pointer_type == InterfacePointer) {
        return static_cast<ScalarInterface *>(ptr)->to_pointer();
    }
    return {p_type, ptr};
}
void Scalar::set_to_zero() {
    if (p_data == nullptr) {
        assert(p_type != nullptr);
        assert(m_constness == IsMutable);
        assert(m_pointer_type == OwnedPointer);
        ScalarPointer::operator=(p_type->allocate(1));
        p_type->assign(to_mut_pointer(), 0, 1);
    }
    // TODO: look at the logic here.
}
scalar_t Scalar::to_scalar_t() const {
    if (p_data == nullptr) {
        return scalar_t(0);
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const ScalarInterface *>(p_data)->as_scalar();
    }
    assert(p_type != nullptr);
    return p_type->to_scalar_t(to_pointer());
}
Scalar Scalar::operator-() const {
    if (p_data == nullptr) {
        return Scalar(p_type);
    }
    if (m_pointer_type == InterfacePointer) {
        return static_cast<const ScalarInterface *>(p_data)->uminus();
    }
    return p_type->uminus(to_pointer());
}

#define RPY_SCALAR_OP(OP, MNAME)                                             \
    Scalar Scalar::operator OP(const Scalar &other) const {                    \
        const ScalarType *type = (p_type != nullptr) ? p_type : other.p_type; \
        if (type == nullptr) {                                                 \
            return Scalar();                                                   \
        }                                                                      \
        return type->MNAME(to_pointer(), other.to_pointer());                  \
    }

RPY_SCALAR_OP(+, add)
RPY_SCALAR_OP(-, sub)
RPY_SCALAR_OP(*, mul)
RPY_SCALAR_OP(/, div)

#undef RPY_SCALAR_OP

#define RPY_SCALAR_IOP(OP, MNAME)                                                         \
    Scalar &Scalar::operator OP(const Scalar &other) {                                     \
        if (m_constness == IsConst) {                                                      \
            throw std::runtime_error("performing inplace operation on const scalar");      \
        }                                                                                  \
                                                                                           \
        if (p_type == nullptr) {                                                           \
            assert(p_data == nullptr);                                                     \
            /* We just established that other.p_data != nullptr */                         \
            assert(other.p_type != nullptr);                                               \
            p_type = other.p_type;                                                         \
        }                                                                                  \
        if (p_data == nullptr) {                                                           \
            if (p_type == nullptr) {                                                       \
                p_type = other.p_type;                                                     \
            }                                                                              \
            set_to_zero();                                                                 \
        }                                                                                  \
        if (m_pointer_type == InterfacePointer) {                                          \
            auto *iface = static_cast<ScalarInterface *>(const_cast<void *>(p_data));     \
            iface->MNAME##_inplace(other);                                                      \
        } else {                                                                           \
            p_type->MNAME##_inplace(to_mut_pointer(), other.to_pointer()); \
        }                                                                                  \
        return *this;                                                                      \
    }

RPY_SCALAR_IOP(+=, add)
RPY_SCALAR_IOP(-=, sub)
RPY_SCALAR_IOP(*=, mul)

Scalar &Scalar::operator/=(const Scalar &other) {
    if (m_constness == IsConst) {
        throw std::runtime_error("performing inplace operation on const scalar");
    }
    if (other.p_data == nullptr) {
        throw std::runtime_error("division by zero");
    }
    if (p_type == nullptr) {
        assert(p_data == nullptr);
        assert(other.p_type != nullptr);
        p_type = other.p_type;
    }
    if (p_data == nullptr) {
        if (p_type == nullptr) {
            p_type = other.p_type->rational_type();
        }
        set_to_zero();
    }
    if (m_pointer_type == InterfacePointer) {
        auto *iface = static_cast<ScalarInterface *>(const_cast<void *>(p_data));
        iface->div_inplace(other);
    } else {
        p_type->rational_type()->div_inplace(to_mut_pointer(), other.to_pointer());
    }
    return *this;
}

#undef RPY_SCALAR_IOP

bool Scalar::operator==(const Scalar &rhs) const noexcept {
    if (p_type == nullptr) {
        return rhs.is_zero();
    }
    return p_type->are_equal(to_pointer(), rhs.to_pointer());
}
bool Scalar::operator!=(const Scalar &rhs) const noexcept {
    return !operator==(rhs);
}
std::ostream &rpy::scalars::operator<<(std::ostream &os, const Scalar &arg) {
    if (arg.type() == nullptr) {
        os << '0';
    } else {
        arg.type()->print(arg.to_pointer(), os);
    }

    return os;
}
