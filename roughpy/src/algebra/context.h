#ifndef RPY_PY_ALGEBRA_CONTEXT_H_
#define RPY_PY_ALGEBRA_CONTEXT_H_

#include "roughpy_module.h"

#include <roughpy/algebra/context.h>


namespace rpy {
namespace python {


class PyContext {
    algebra::context_pointer p_ctx;


public:

    PyContext(algebra::context_pointer ctx) : p_ctx(std::move(ctx))
    {}

    algebra::context_pointer to_context() && noexcept { return p_ctx; }
    algebra::context_pointer get_context() const noexcept { return p_ctx; }

    const algebra::Context& operator*() const noexcept { return *p_ctx; }
    const algebra::Context* operator->() const noexcept { return p_ctx.get(); }

};

void init_context(py::module_& m);


} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_CONTEXT_H_
