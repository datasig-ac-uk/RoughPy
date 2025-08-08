#ifndef ROUGHPY_COMPUTE__SRC_LIE_BASIS_H
#define ROUGHPY_COMPUTE__SRC_LIE_BASIS_H


#include "py_headers.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct _PyLieBasis
{
    PyObject_HEAD
    int32_t width;
    int32_t depth;
    PyObject* degree_begin;
    PyObject* data;
} PyLieBasis;


extern PyTypeObject PyLieBasis_Type;


int init_lie_basis(PyObject* module);

#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_LIE_BASIS_H