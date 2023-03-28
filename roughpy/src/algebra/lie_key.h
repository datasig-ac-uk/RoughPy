#ifndef RPY_PY_ALGEBRA_LIE_KEY_H_
#define RPY_PY_ALGEBRA_LIE_KEY_H_

#include "roughpy_module.h"

#include <boost/container/small_vector.hpp>

#include <roughpy/algebra/context_fwd.h>
#include <roughpy/algebra/lie_basis.h>

#include "lie_letter.h"



namespace rpy {
namespace python {

class PyLieKey {
public:
    using container_type = boost::container::small_vector<PyLieLetter, 2>;

private:
    container_type m_data;
    deg_t m_width;

public:
    explicit PyLieKey(deg_t width);
    explicit PyLieKey(deg_t width, let_t letter);
    explicit PyLieKey(deg_t width, const boost::container::small_vector_base<PyLieLetter> &data);
    explicit PyLieKey(deg_t width, let_t left, let_t right);
    explicit PyLieKey(deg_t width, let_t left, const PyLieKey &right);
    explicit PyLieKey(deg_t width, const PyLieKey &left, const PyLieKey &right);
    explicit PyLieKey(algebra::LieBasis basis, key_type key);
    PyLieKey(const algebra::Context *ctx, key_type key);

    deg_t width() const noexcept { return m_width; }

    bool is_letter() const noexcept;
    let_t as_letter() const;
    std::string to_string() const;
    PyLieKey lparent() const;
    PyLieKey rparent() const;

    deg_t degree() const;

    bool equals(const PyLieKey &other) const noexcept;
};

void init_py_lie_key(py::module_& m);

} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_LIE_KEY_H_
