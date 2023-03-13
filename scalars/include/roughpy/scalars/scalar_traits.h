#ifndef ROUGHPY_SCALARS_SCALAR_TRAITS_H_
#define ROUGHPY_SCALARS_SCALAR_TRAITS_H_

#include "scalar.h"
#include "scalars_fwd.h"

namespace rpy {
namespace scalars {

template <typename T>
class scalar_type_trait {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;

    static const ScalarType *get_type() noexcept {
        return ScalarType::of<T>();
    }

    static Scalar make(value_type &&arg) {
        return Scalar(get_type(), std::move(arg));
    }
};

template <typename T>
class scalar_type_trait<T &> {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;

    static const ScalarType *get_type() noexcept {
        return scalar_type_trait<T>::get_type();
    }

    static Scalar make(reference arg) {
        return Scalar(ScalarPointer(get_type(), &arg));
    }
};

template <typename T>
class scalar_type_trait<const T &> {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;

    static const ScalarType *get_type() noexcept {
        return scalar_type_trait<T>::get_type();
    }

    static Scalar make(const_reference arg) {
        return Scalar(ScalarPointer(get_type(), &arg));
    }
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_TRAITS_H_
