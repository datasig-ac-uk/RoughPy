#define NPY_IMPORT_THE_APIS_PLEASE
#include <roughpy/pycore/py_headers.h>

#include "dense_basic.h"
#include "dense_intermediate.h"
#include "lie_basis.h"
#include "sparse_matrix.h"
#include "tensor_basis.h"



static PyMethodDef roughpy_compute_methods[] = {
        {             "dense_ft_fma",
         (PyCFunction) py_dense_ft_fma,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor fused multiply-add."                                 },
        {     "dense_ft_inplace_mul",
         (PyCFunction) py_dense_ft_inplace_mul,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor inplace multiply."                                   },
        {        "dense_ft_antipode",
         (PyCFunction) py_dense_antipode,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor antipode"                                            },
        {             "dense_st_fma",
         (PyCFunction) py_dense_st_fma,
         METH_VARARGS | METH_KEYWORDS,
         "dense shuffle tensor fma"                                              },
        {             "dense_ft_exp",
         (PyCFunction) py_dense_ft_exp,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor exponential"                                         },
        {           "dense_ft_fmexp",
         (PyCFunction) py_dense_ft_fmexp,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor fused-multiply exponential"                          },

        {             "dense_ft_log",
         (PyCFunction) py_dense_ft_log,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor logarithm"                                           },
        {"dense_ft_adjoint_left_mul",
         (PyCFunction) py_dense_ft_adj_lmul,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor adjoint left multiply"                               },
        {      "dense_lie_to_tensor",
         (PyCFunction) py_dense_lie_to_tensor,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor Lie to tensor map"                                   },
        {      "dense_tensor_to_lie",
         (PyCFunction) py_dense_tensor_to_lie,
         METH_VARARGS | METH_KEYWORDS,
         "dense free tensor tensor to Lie map"                                   },
        {          "check_lie_basis_data",
         (PyCFunction) PyLieBasis_check_data,
         METH_VARARGS | METH_KEYWORDS,
         "check if data and degree_begin data form a valid Lie basis"            },
        {                       NULL,     NULL, 0,                           NULL}
};

static PyModuleDef_Slot roughpy_compute_slots[] = {
    {Py_mod_exec, init_lie_basis},
    {Py_mod_exec, init_tensor_basis},
    {Py_mod_exec, init_sparse_matrix},
#if PY_VERSION_HEX >= PYVER_HEX(3, 12)
        {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
        {                          0,                                       NULL}
};

static PyModuleDef rpy_compute_internals_module
        = {.m_base = PyModuleDef_HEAD_INIT,
           .m_name = "_rpy_compute_internals",
           .m_size = 0,
           .m_slots = roughpy_compute_slots,
           .m_methods = roughpy_compute_methods};

PyMODINIT_FUNC PyInit__rpy_compute_internals(void)
{
    import_array();

    return PyModuleDef_Init(&rpy_compute_internals_module);
}