#ifndef ROUGHPY_COMPUTE__SRC_DENSE_BASIC_H
#define ROUGHPY_COMPUTE__SRC_DENSE_BASIC_H


#include <roughpy/pycore/py_headers.h>

#ifdef __cplusplus
extern "C" {
#endif


RPY_NO_EXPORT
PyObject* py_dense_antipode(PyObject*, PyObject*, PyObject*);

RPY_NO_EXPORT
PyObject* py_dense_ft_fma(PyObject*, PyObject*, PyObject*);
RPY_NO_EXPORT
PyObject* py_dense_ft_inplace_mul(PyObject*, PyObject*, PyObject*);


RPY_NO_EXPORT
PyObject* py_dense_ft_adj_lmul(PyObject*, PyObject*, PyObject*);

RPY_NO_EXPORT
PyObject* py_dense_st_fma(PyObject* , PyObject* , PyObject* );


RPY_NO_EXPORT
PyObject* py_dense_lie_to_tensor(PyObject* , PyObject* , PyObject*);
RPY_NO_EXPORT
PyObject* py_dense_tensor_to_lie(PyObject* , PyObject* , PyObject*);




#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_DENSE_BASIC_H
