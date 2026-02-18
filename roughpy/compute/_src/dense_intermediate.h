#ifndef ROUGHPY_COMPUTE__SRC_DENSE_INTERMEDIATE_H
#define ROUGHPY_COMPUTE__SRC_DENSE_INTERMEDIATE_H

#include <roughpy/pycore/py_headers.h>

#ifdef __cplusplus
extern "C" {
#endif

RPY_NO_EXPORT
PyObject* py_dense_ft_exp(PyObject*, PyObject*, PyObject*);

RPY_NO_EXPORT
PyObject* py_dense_ft_fmexp(PyObject*, PyObject*, PyObject*);

RPY_NO_EXPORT
PyObject* py_dense_ft_log(PyObject*, PyObject*, PyObject*);



#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_DENSE_INTERMEDIATE_H
