


#define NPY_IMPORT_THE_APIS_PLEASE
#include "py_headers.h"






static PyModuleDef rpy_compute_internals_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_rpy_compute_internals",
    .m_size = 0,
};


PyMODINIT_FUNC
PyInit__rpy_compute_internals(void)
{
    import_array();

    PyObject *module = PyModule_Create(&rpy_compute_internals_module);
    if (module == nullptr) {
        return nullptr;
    }

    return module;
}