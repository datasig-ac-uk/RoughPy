#ifndef ROUGHPY_SCALARS_SCALAR_H_
#define ROUGHPY_SCALARS_SCALAR_H_

#include "scalars_fwd.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "scalar_interface.h"
#include "scalar_pointer.h"
#include "scalar_type.h"

namespace rpy {
namespace scalars {

class Scalar : private ScalarPointer {
    using ScalarPointer::IsConst;
    using ScalarPointer::IsMutable;
    using ScalarPointer::m_constness;

public:

    enum PointerType {
        OwnedPointer,
        BorrowedPointer,
        InterfacePointer
    };

private:

    PointerType m_pointer_type = OwnedPointer;


public:
    Scalar() = default;
    Scalar(const Scalar &other);
    Scalar(Scalar &&other) noexcept;

    explicit Scalar(const ScalarType *type);
    explicit Scalar(scalar_t arg);
    explicit Scalar(ScalarPointer ptr);
    explicit Scalar(ScalarInterface *interface_ptr);
    Scalar(ScalarPointer ptr, PointerType ptype);

    Scalar(const ScalarType *type, scalar_t arg);

    template <typename I, typename J, typename = std::enable_if_t<std::is_integral<I>::value && std::is_integral<J>::value>>
    Scalar(const ScalarType *type, I numerator, J denominator) {
        if (type == nullptr) {
            type = get_type("rational");
        }
        ScalarPointer::operator=(type->allocate(1));
        type->assign(static_cast<const ScalarPointer &>(*this),
                     static_cast<long long>(numerator),
                     static_cast<long long>(denominator));
    }

    template <typename ScalarArg>
    Scalar(const ScalarType *type, ScalarArg arg) {
        const auto *scalar_arg_type = ScalarType::of<ScalarArg>();
        if (scalar_arg_type != nullptr) {
            if (type == nullptr) {
                type = scalar_arg_type;
            }
            ScalarPointer::operator=(type->allocate(1));
            type->convert_copy(to_mut_pointer(), {scalar_arg_type, std::addressof(arg)}, 1);
        } else {
            const auto& id = type_id_of<ScalarArg>();
            if (type == nullptr) {
                type = ScalarType::for_id(id);
            }
            ScalarPointer::operator=(type->allocate(1));
            type->convert_copy(to_mut_pointer(), &arg, 1, id);
        }
    }

    ~Scalar();

    Scalar &operator=(const Scalar &other);
    Scalar &operator=(Scalar &&other) noexcept;

    template <typename T, typename = std::enable_if_t<!std::is_same<Scalar, std::decay_t<T>>::value>>
    Scalar &operator=(T arg) {
        if (p_type == nullptr) {
            p_type = ScalarType::of<std::decay_t<T>>();
        } else {
            if (m_constness == IsConst) {
                throw std::runtime_error("attempting to assign value to const scalar");
            }
        }

        if (p_data == nullptr) {
            m_pointer_type = OwnedPointer;
            ScalarPointer::operator=(p_type->allocate(1));
        }

        const auto &type_id = type_id_of<T>();
        if (m_pointer_type == InterfacePointer) {
            static_cast<ScalarInterface *>(const_cast<void *>(p_data))
                ->assign(std::addressof(arg), type_id);
        } else {
            p_type->convert_copy(static_cast<const ScalarPointer &>(*this),
                                 std::addressof(arg), 1, type_id);
        }

        return *this;
    }

    using ScalarPointer::is_const;
    using ScalarPointer::type;
    bool is_value() const noexcept;
    bool is_zero() const noexcept;

    ScalarPointer to_pointer() const noexcept;
    ScalarPointer to_mut_pointer();

    void set_to_zero();

    scalar_t to_scalar_t() const;

    Scalar operator-() const;

    Scalar operator+(const Scalar &other) const;
    Scalar operator-(const Scalar &other) const;
    Scalar operator*(const Scalar &other) const;
    Scalar operator/(const Scalar &other) const;

    Scalar &operator+=(const Scalar &other);
    Scalar &operator-=(const Scalar &other);
    Scalar &operator*=(const Scalar &other);
    Scalar &operator/=(const Scalar &other);

    bool operator==(const Scalar &other) const noexcept;
    bool operator!=(const Scalar &other) const noexcept;


};

ROUGHPY_SCALARS_EXPORT
std::ostream &operator<<(std::ostream &, const Scalar& arg);

namespace dtl {

template <typename T>
struct type_of_T_defined {
    static T cast(ScalarPointer scalar) {
        const auto *tp = ScalarType::of<T>();
        if (tp == scalar.type()) {
            return *scalar.raw_cast<const T *>();
        }
        if (tp == scalar.type()->rational_type()) {
            return *scalar.raw_cast<const T *>();
        }

        T result;
        ScalarPointer dst(tp, &result);
        tp->convert_copy(dst, scalar, 1);
        return result;
    }
};

template <typename T>
struct type_of_T_not_defined {
    static T cast(ScalarPointer scalar) {
        T result;
        scalar.type()->convert_copy({nullptr, &result}, scalar.ptr(), 1, type_id_of<T>());
        return result;
    }
};

}


template <typename T>
inline traits::remove_cv_ref_t<T> scalar_cast(const Scalar& scalar) {
    if (scalar.is_zero()) {
        return T(0);
    }
    using bare_t = traits::remove_cv_ref_t<T>;
    using impl_t = traits::detected_or_t<dtl::type_of_T_not_defined<bare_t>,
        dtl::type_of_T_defined, bare_t>;

    // Now we are sure that scalar.type() != nullptr
    // and scalar.ptr() != nullptr
    return impl_t::cast(scalar.to_pointer());
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_H_
