#define NPY_IMPORT_THE_APIS_PLEASE
#include "py_headers.h"


#include "tensor_basis.h"
#include "lie_basis.h"
#include "dense_basic.h"
#include "dense_intermediate.h"


static int init_module(PyObject* module)
{
    if (init_lie_basis(module) < 0) { return -1; }
    if (init_tensor_basis(module) < 0) { return -1; }

    return 0;
}


static PyMethodDef roughpy_compute_methods[] = {
        {
                "dense_ft_fma", (PyCFunction) py_dense_ft_fma,
                METH_VARARGS | METH_KEYWORDS,
                "dense free tensor fused multiply-add."
        },
        {
                "dense_ft_inplace_mul", (PyCFunction) py_dense_ft_inplace_mul,
                METH_VARARGS | METH_KEYWORDS,
                "dense free tensor inplace multiply."
        },
        {
                "dense_ft_antipode", (PyCFunction) py_dense_antipode,
                METH_VARARGS | METH_KEYWORDS,
                "dense free tensor antipode"
        },
        {
                "dense_ft_exp", (PyCFunction) py_dense_ft_exp,
                METH_VARARGS | METH_KEYWORDS,
                "dense free tensor exponential"
        },
        {
                "dense_ft_fmexp", (PyCFunction) py_dense_ft_fmexp,
                METH_VARARGS | METH_KEYWORDS,
                "dense free tensor fused-multiply exponential"
        },

        {
                "dense_ft_log", (PyCFunction) py_dense_ft_log,
                METH_VARARGS | METH_KEYWORDS,
                "dense free tensor logarithm"
        },
        {NULL, NULL, 0, NULL}
};


static PyModuleDef_Slot roughpy_compute_slots[] = {
        {Py_mod_exec, init_module},
#if Py_VERSION_HEX >= PYVER_HEX(3, 12)
        {
                Py_mod_multipe_interpreters,
                Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED
        },
#endif
        {0, NULL}
};


static PyModuleDef rpy_compute_internals_module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "_rpy_compute_internals",
        .m_size = 0,
        .m_slots = roughpy_compute_slots,
        .m_methods = roughpy_compute_methods
};


PyMODINIT_FUNC
PyInit__rpy_compute_internals(void)
{
    import_array();

    return PyModuleDef_Init(&rpy_compute_internals_module);
}