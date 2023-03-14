#ifndef RPY_PY_ALGEBRA_CONTEXT_H_
#define RPY_PY_ALGEBRA_CONTEXT_H_

#include <pybind11/pybind11.h>

#include <roughpy/algebra/context.h>


namespace rpy {
namespace python {

namespace py = pybind11;

class PyContext {
    algebra::context_pointer p_ctx;


public:

    algebra::context_pointer to_context() && noexcept { return p_ctx; }
};


} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_CONTEXT_H_
