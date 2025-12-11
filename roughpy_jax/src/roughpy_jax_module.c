// filepath: /workspaces/RoughPy/roughpy_jax/src/rpy_jax__module.cpp
// Minimal CPython extension module for roughpy_jax._rpy_jax_internals.
// This does not expose any Python-callable functions yet; the JAX FFI
// still uses ctypes to load and register the raw symbols.

#define PY_SSIZE_T_CLEAN
#include <Python.h> 
#include <xla/ffi/api/c_api.h>

#include "cpu/dense_ft_antipode.h"
#include "cpu/dense_ft_exp.h"
#include "cpu/dense_ft_fma.h"
#include "cpu/dense_ft_fmexp.h"
#include "cpu/dense_ft_log.h"
#include "cpu/dense_st_fma.h"


static inline int add_fn_capsule(PyObject* dict, const char* name, void* fn_ptr)
{
    PyObject* capsule = PyCapsule_New(fn_ptr, name, NULL);
    if (capsule == NULL) {
        return -1;
    }

    const int ret = PyDict_SetItemString(dict, name, capsule);
    Py_DECREF(capsule);

    return ret;
}

#define RPJ_ADD_FN_CAPSULE(dict, fn) add_fn_capsule(dict, #fn, (void*) fn)


static int make_jax_function_dict(PyObject* module) {
   int ret = -1;
   PyObject* dict = PyDict_New();
   if (dict == NULL) { return ret; }

   if (RPJ_ADD_FN_CAPSULE(dict, cpu_dense_ft_antipode) < 0) {
       goto finish;
   }
   if (RPJ_ADD_FN_CAPSULE(dict, cpu_dense_ft_exp) < 0) {
       goto finish;
   }
    if (RPJ_ADD_FN_CAPSULE(dict, cpu_dense_ft_fma) < 0) {
         goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cpu_dense_ft_fmexp) < 0) {
         goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cpu_dense_ft_log) < 0) {
         goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cpu_dense_st_fma) < 0) {
         goto finish;
    }


   ret = PyModule_AddObjectRef(module, "cpu_functions", dict); // possibly needs the compat header
finish:
   Py_DECREF(dict);
   return ret;
}

static PyMethodDef rpy_jax_methods[] = {
    {NULL, NULL, 0, NULL}
};

static PyModuleDef_Slot rpy_jax_slots[] = {
    {Py_mod_exec, make_jax_function_dict}, // Needed Py_mod_exec as init doesn't exist 
#if PY_VERSION_HEX >= 0x030C0000  /* Python 3.12.0+ */
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
    {0, NULL}
};

static struct PyModuleDef rpy_jax_internals_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_rpy_jax_internals",  
    .m_doc = "RoughPy JAX internals",
    .m_size = 0,                     
    .m_methods = rpy_jax_methods,    
    .m_slots = rpy_jax_slots,               
};

PyMODINIT_FUNC PyInit__rpy_jax_internals(void)
{
    return PyModuleDef_Init(&rpy_jax_internals_module);
}
