#ifndef RPY_PY_ALGEBRA_LIE_LETTER_H_
#define RPY_PY_ALGEBRA_LIE_LETTER_H_

#include "roughpy_module.h"

#include <iosfwd>

namespace rpy {
namespace python {

class PyLieLetter {
    dimn_t m_data = 0;

    constexpr explicit PyLieLetter(dimn_t raw) : m_data(raw) {}

public:
    PyLieLetter() = default;

    static constexpr PyLieLetter from_letter(let_t letter) {
        return PyLieLetter(1 + (dimn_t(letter) << 1));
    }

    static constexpr PyLieLetter from_offset(dimn_t offset) {
        return PyLieLetter(offset << 1);
    }

    constexpr bool is_offset() const noexcept {
        return (m_data & 1) == 0;
    }

    explicit operator let_t() const noexcept {
        return let_t(m_data >> 1);
    }

    explicit constexpr operator dimn_t() const noexcept {
        return m_data >> 1;
    }

    friend std::ostream &operator<<(std::ostream &os, const PyLieLetter &let);
};

std::ostream& operator<<(std::ostream& os, const PyLieLetter& let);


} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_LIE_LETTER_H_
