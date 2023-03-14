#ifndef RPY_PY_ALGEBRA_LIE_KEY_ITERATOR_H_
#define RPY_PY_ALGEBRA_LIE_KEY_ITERATOR_H_

#include <limits>

#include "roughpy_module.h"

#include <roughpy/algebra/context_fwd.h>

#include "lie_key.h"
#include "context.h"

namespace rpy {
namespace python {

class PyLieKeyIterator {
    key_type m_current;
    key_type m_end;
    algebra::context_pointer p_ctx;

public:

    explicit PyLieKeyIterator(const PyContext& ctx,
                     key_type current=1,
                     key_type end=std::numeric_limits<key_type>::max());

    PyLieKey next();

};


void init_lie_key_iterator(py::module_& m);

} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_LIE_KEY_ITERATOR_H_
