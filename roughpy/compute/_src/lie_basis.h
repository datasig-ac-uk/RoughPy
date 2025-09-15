#ifndef ROUGHPY_COMPUTE__SRC_LIE_BASIS_H
#define ROUGHPY_COMPUTE__SRC_LIE_BASIS_H


#include "py_headers.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct _PyLieBasis PyLieBasis;


extern PyTypeObject PyLieBasis_Type;


PyObject* get_l2t_matrix(PyObject* basis, PyObject* dtype);
PyObject* get_t2l_matrix(PyObject* basis, PyObject* dtype);

int init_lie_basis(PyObject* module);

#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_LIE_BASIS_H