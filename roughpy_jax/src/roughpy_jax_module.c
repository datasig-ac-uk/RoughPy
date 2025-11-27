// filepath: /workspaces/RoughPy/roughpy_jax/src/rpy_jax_internals_module.cpp
// Minimal CPython extension module for roughpy_jax._rpy_jax_internals.
// This does not expose any Python-callable functions yet; the JAX FFI
// still uses ctypes to load and register the raw symbols.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include 

static PyMethodDef rpy_jax_methods[] = {
    {NULL, NULL, 0, NULL}
};

static PyModuleDef_Slot rpy_jax_slots[] = {
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
    // .m_traverse = nullptr,        
    // .m_clear = nullptr,           
    // .m_free = nullptr            
};

PyMODINIT_FUNC PyInit__rpy_jax_internals(void)
{
    return PyModuleDef_Init(&rpy_jax_internals_module);
}
