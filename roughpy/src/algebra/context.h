#ifndef RPY_PY_ALGEBRA_CONTEXT_H_
#define RPY_PY_ALGEBRA_CONTEXT_H_

#include "roughpy_module.h"

#include <roughpy/algebra/context.h>


namespace rpy {
namespace python {


extern "C" {

struct RPyContext {
    PyObject_VAR_HEAD
        algebra::context_pointer p_ctx;
};

extern PyTypeObject RPyContext_Type;

PyObject* RPyContext_FromContext(algebra::context_pointer ctx);

}

inline const algebra::context_pointer &ctx_cast(PyObject *ctx) {
    assert(ctx != nullptr && Py_TYPE(ctx) == &python::RPyContext_Type);
    return reinterpret_cast<RPyContext *>(ctx)->p_ctx;
}

void init_context(py::module_& m);


} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_CONTEXT_H_
