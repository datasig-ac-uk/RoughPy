#ifndef ROUGHPY_COMPUTE__SRC_SPARSE_MATRIX_H
#define ROUGHPY_COMPUTE__SRC_SPARSE_MATRIX_H

#include "py_headers.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct _PySparseMatrix {
    PyObject_HEAD
    PyObject* data;
    PyObject* indices;
    PyObject* indptr;
    npy_intp rows;
    npy_intp cols;
} PySparseMatrix;


typedef struct _SMHFrame
{
    char* data;
    npy_intp* indices;
    npy_intp size;
} SMHFrame;

typedef struct _SMHelper
{
    SMHFrame* frames;
    PyArrayObject* data;
    PyArrayObject* indices;
    PyArrayObject* indptr;
    npy_intp size;
    npy_intp alloc;
    npy_intp nnz;
} SMHelper;





/**
 * @brief Create a new sparse matrix from the components
 *
 * Steals a reference to each of its arguments.
 */
PyObject*
py_sparse_matrix_from_components(
    PyObject* data,
    PyObject* indices,
    PyObject* indptr,
    npy_intp nrows,
    npy_intp ncols
    );



int init_sparse_matrix(PyObject* module);


int smh_init(SMHelper* helper, PyArray_Descr* dtype, npy_intp alloc, npy_intp nnz_est);
void smh_free(SMHelper* helper);
int smh_insert_frame(SMHelper* helper);
void* smh_get_scalar_for_index(SMHelper* helper, npy_intp index);

PyObject* smh_build_matrix(SMHelper* helper, npy_intp nrows, npy_intp ncols);


static inline SMHFrame* smh_current_frame(SMHelper* helper)
{
    assert(helper->size > 0);
    return &helper->frames[helper->size - 1];
}

static inline int smh_dtype(SMHelper* helper)
{
    return PyArray_TYPE(helper->data);
}


static inline void insert_zero(void* ptr, int typenum)
{
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (typenum) {
        case NPY_FLOAT: *(npy_float*) ptr = 0.0f;
            break;
        case NPY_DOUBLE: *(npy_double*) ptr = 0.0;
            break;
        case NPY_LONGDOUBLE: *(npy_longdouble*) ptr = 0.0L;
    }
}


static inline void insert_one(void* ptr, int typenum)
{
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (typenum) {
        case NPY_FLOAT: *(npy_float*) ptr = 1.0f;
            break;
        case NPY_DOUBLE: *(npy_double*) ptr = 1.0;
            break;
        case NPY_LONGDOUBLE: *(npy_longdouble*) ptr = 1.0L;
    }
}


#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_SPARSE_MATRIX_H