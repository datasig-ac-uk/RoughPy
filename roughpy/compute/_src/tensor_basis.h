#ifndef ROUGHPY_COMPUTE__SRC_TENSOR_BASIS_H
#define ROUGHPY_COMPUTE__SRC_TENSOR_BASIS_H


#include <roughpy/pycore/py_headers.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef struct _PyTensorBasis PyTensorBasis;

extern PyTypeObject* PyTensorBasis_Type;


int init_tensor_basis(PyObject* module);

static inline int PyTensorBasis_Check(PyObject* obj)
{
  return PyObject_TypeCheck(obj, PyTensorBasis_Type);
}

PyTensorBasis* PyTensorBasis_get(int32_t width, int32_t depth);

int32_t PyTensorBasis_width(PyTensorBasis* basis);
int32_t PyTensorBasis_depth(PyTensorBasis* basis);
PyArrayObject* PyTensorBasis_degree_begin(PyTensorBasis* basis);


#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_TENSOR_BASIS_H