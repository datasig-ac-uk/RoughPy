#ifndef RPY_PY_ALGEBRA_TENSOR_KEY_H_
#define RPY_PY_ALGEBRA_TENSOR_KEY_H_

#include "roughpy_module.h"

namespace rpy {
namespace python {
namespace maths {
template <typename I, typename E>
constexpr I power(I arg, E exponent) noexcept {
    if (exponent == 0) {
        return I(1);
    }
    if (exponent == 1) {
        return arg;
    }
    auto recurse = power(arg, exponent / 2);
    return recurse * recurse * ((exponent & 1) == 1 ? arg : I(1));
}

template <typename I, typename B>
constexpr I log(I arg, B base) noexcept {
    return (arg < base) ? I(0) : I(1) + log(arg / static_cast<I>(base), base);
}

}// namespace maths

class PyTensorKey {
    key_type m_key;
    deg_t m_width;
    deg_t m_depth;

public:
    explicit PyTensorKey(key_type key, deg_t width, deg_t depth);

    explicit operator key_type() const noexcept;

    std::string to_string() const;
    PyTensorKey lparent() const;
    PyTensorKey rparent() const;
    bool is_letter() const;

    deg_t width() const;
    deg_t depth() const;

    deg_t degree() const;
    std::vector<let_t> to_letters() const;

    bool equals(const PyTensorKey &other) const noexcept;
    bool less(const PyTensorKey &other) const noexcept;
};

void init_py_tensor_key(py::module_& m);

} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_TENSOR_KEY_H_
