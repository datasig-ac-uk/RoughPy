#ifndef RPY_PY_ALGEBRA_CONTEXT_H_
#define RPY_PY_ALGEBRA_CONTEXT_H_

#include "roughpy_module.h"

#include <roughpy/algebra/context.h>


namespace rpy {
namespace python {


class PyContext {
    algebra::context_pointer p_ctx;


public:

    algebra::context_pointer to_context() && noexcept { return p_ctx; }
};


} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_CONTEXT_H_
