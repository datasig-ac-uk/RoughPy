#ifndef ROUGHPY_COMPUTE__SRC_SPARSE_MATRIX_H
#define ROUGHPY_COMPUTE__SRC_SPARSE_MATRIX_H

#include <roughpy/pycore/py_headers.h>


#ifdef __cplusplus
extern "C" {

#endif

enum SMFormat {
    SM_CSR = 0,
    SM_CSC = 1,
};


typedef struct _PySparseMatrix {
    PyObject_HEAD
    PyObject *data;
    PyObject *indices;
    PyObject *indptr;
    npy_intp rows;
    npy_intp cols;
    enum SMFormat format;
} PySparseMatrix;


typedef struct _SMHFrame {
    char *data;
    npy_intp *indices;
    npy_intp size;
} SMHFrame;



typedef struct _SMHelper {
    SMHFrame *frames;
    PyArrayObject *data;
    PyArrayObject *indices;
    PyArrayObject *indptr;
    npy_intp size;
    npy_intp alloc;
    npy_intp nnz;
    npy_intp rows;
    npy_intp cols;
    int flags;
} SMHelper;


/**
 * @brief Create a new sparse matrix from the components
 *
 * Steals a reference to each of its arguments.
 */
PyObject *
py_sparse_matrix_from_components(
    PyObject *data,
    PyObject *indices,
    PyObject *indptr,
    npy_intp nrows,
    npy_intp ncols
);


RPY_NO_EXPORT
int init_sparse_matrix(PyObject *module);

extern PyTypeObject* PySparseMatrix_Type;


static inline int PySparseMatrix_Check(PyObject* obj)
{
    return PyObject_TypeCheck(obj, PySparseMatrix_Type);
}


RPY_NO_EXPORT
int smh_init(SMHelper *helper, PyArray_Descr *dtype,
             npy_intp nrows, npy_intp ncols,
             npy_intp nnz_est, int format);

RPY_NO_EXPORT
void smh_free(SMHelper *helper);

RPY_NO_EXPORT
int smh_insert_frame(SMHelper *helper);

RPY_NO_EXPORT
void *smh_get_scalar_for_index(SMHelper *helper, npy_intp index);

RPY_NO_EXPORT
int smh_insert_value_at_index(SMHelper* helper, npy_intp index, const void* value);

// RPY_NO_EXPORT
// int smh_swap_format(SMHelper *helper);

RPY_NO_EXPORT
PyObject *smh_build_matrix(SMHelper *helper);


static inline SMHFrame *smh_current_frame(SMHelper *helper) {
    assert(helper->size > 0);
    return &helper->frames[helper->size - 1];
}

static inline int smh_dtype(SMHelper *helper) {
    return PyArray_TYPE(helper->data);
}

static inline void insert_zero(void *ptr, int typenum) {
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (typenum)
    {
        case NPY_FLOAT: *(npy_float *) ptr = 0.0f;
            break;
        case NPY_DOUBLE: *(npy_double *) ptr = 0.0;
            break;
        case NPY_LONGDOUBLE: *(npy_longdouble *) ptr = 0.0L;
    }
}

static inline int smh_is_csr(SMHelper* helper)
{
    return helper->flags & SM_CSR;
}

static inline int smh_is_csc(SMHelper* helper)
{
    return helper->flags & SM_CSC;
}

static inline void insert_one(void *ptr, int typenum) {
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (typenum)
    {
        case NPY_FLOAT: *(npy_float *) ptr = 1.0f;
            break;
        case NPY_DOUBLE: *(npy_double *) ptr = 1.0;
            break;
        case NPY_LONGDOUBLE: *(npy_longdouble *) ptr = 1.0L;
    }
}


#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_SPARSE_MATRIX_H
