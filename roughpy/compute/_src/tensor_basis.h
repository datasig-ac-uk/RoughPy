#ifndef ROUGHPY_COMPUTE__SRC_TENSOR_BASIS_H
#define ROUGHPY_COMPUTE__SRC_TENSOR_BASIS_H


#include "py_headers.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct _PyTensorBasis
{
    PyObject_HEAD
    int32_t width;
    int32_t depth;
    PyObject* degree_begin;
} PyTensorBasis;

extern PyTypeObject PyTensorBasis_Type;


int init_tensor_basis(PyObject* module);


#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_TENSOR_BASIS_H