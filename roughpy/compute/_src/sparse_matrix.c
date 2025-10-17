// ReSharper disable CppParameterMayBeConstPtrOrRef
#include "sparse_matrix.h"


#include <stdlib.h>
#include <string.h>
#include <stddef.h>



#define RPC_PYCOMPAT_INCLUDE_STRUCTMEMBER 1
#include "py_compat.h"

#define SMH_FLAGS_FORMAT_MASK 0x7;



static PyObject* sparse_matrix_new(
    PyTypeObject* type,
    PyObject* Py_UNUSED(args),
    PyObject* Py_UNUSED(kwargs))
{
    PySparseMatrix* self = (PySparseMatrix*) type->tp_alloc(type, 0);
    if (!self) { return NULL; }

    Py_INCREF(Py_None);
    Py_XSETREF(self->data, Py_None);
    Py_INCREF(Py_None);
    Py_XSETREF(self->indices, Py_None);
    Py_INCREF(Py_None);
    Py_XSETREF(self->indptr, Py_None);
    self->rows = 0;
    self->cols = 0;

    return (PyObject*) self;
}


static void sparse_matrix_dealloc(PySparseMatrix* self)
{
    Py_XDECREF(self->data);
    Py_XDECREF(self->indices);
    Py_XDECREF(self->indptr);
    Py_TYPE(self)->tp_free((PyObject*) self);
}


static int sparse_matrix_init(PySparseMatrix* self,
                              PyObject* args,
                              PyObject* kwargs)
{
    static char* kwords[] = {"data", "indices", "indptr", "rows", "cols", NULL};
    PyObject* data = NULL;
    PyObject* indices = NULL;
    PyObject* indptr = NULL;
    int rows = 0;
    int cols = 0;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOOii",
                                     kwords,
                                     &data,
                                     &indices,
                                     &indptr,
                                     &rows,
                                     &cols)) { return -1; }

    Py_INCREF(data);
    Py_SETREF(self->data, data);
    Py_INCREF(indices);
    Py_SETREF(self->indices, indices);
    Py_INCREF(indptr);
    Py_SETREF(self->indptr, indptr);
    self->rows = rows;
    self->cols = cols;

    return 0;
}


PyMemberDef PySparseMatrix_members[] = {
        {"data", Py_T_OBJECT_EX, offsetof(PySparseMatrix, data), 0, "data"},
        {"indices", Py_T_OBJECT_EX, offsetof(PySparseMatrix, indices), 0,
         "indices"},
        {"indptr", Py_T_OBJECT_EX, offsetof(PySparseMatrix, indptr), 0,
         "indptr"},
        {"rows", Py_T_PYSSIZET, offsetof(PySparseMatrix, rows), READONLY,
         "rows"},
        {"cols", Py_T_PYSSIZET, offsetof(PySparseMatrix, cols), READONLY,
         "cols"},
        {NULL}
};

static PyObject* get_shape(PyObject* obj)
{
    PySparseMatrix* self = (PySparseMatrix*) obj;
    return Py_BuildValue("(ii)", self->rows, self->cols);
}

static PyObject* get_nnz(PyObject* obj)
{
    PySparseMatrix* self = (PySparseMatrix*) obj;
    npy_intp nnz = PyArray_SHAPE((PyArrayObject*) self->data)[0];
    return PyLong_FromSsize_t(nnz);
}

static PyObject* get_dtype(PyObject* obj)
{
    PySparseMatrix* self = (PySparseMatrix*) obj;
    PyObject* dtype = (PyObject*) PyArray_DESCR((PyArrayObject*) self->data);
    Py_INCREF(dtype);
    return dtype;
}

static PyObject* get_ndim(PyObject* obj)
{
    return PyLong_FromLong(2);
}

static PyObject* get_format(PyObject* obj)
{
    PySparseMatrix* self = (PySparseMatrix*) obj;
    switch (self->format) {
        case SM_CSC:
            return PyUnicode_FromString("csc");
        case SM_CSR:
            return PyUnicode_FromString("csr");
    }
    PyErr_SetString(PyExc_ValueError, "unknown format");
    return NULL;
}


PyGetSetDef PySparseMatrix_getsets[] = {
    {"shape", (getter) get_shape, NULL, "the shape of the matrix", NULL},
    {"nnz", (getter) get_nnz, NULL, "the number of non-zero entries in the matrix", NULL},
    {"dtype", (getter) get_dtype, NULL, "the dtype of the matrix", NULL},
    {"ndim", (getter) get_ndim, NULL, "the ndim of the matrix, is always 2", NULL},
    {"format", (getter) get_format, NULL, "the format of the matrix, either 'csc' or 'csr'", NULL},
    {NULL}
};

PyMethodDef PySparseMatrix_methods[] = {
    {"getformat", (PyCFunction) get_format, METH_O, "get the string representation of the matrix format"},
    {NULL}
};

PyTypeObject PySparseMatrix_Type = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = RPY_CPT_TYPE_NAME(SparseMatrix),
        .tp_basicsize = sizeof(PySparseMatrix),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "SparseMatrix",
        .tp_new = (newfunc) sparse_matrix_new,
        .tp_dealloc = (destructor) sparse_matrix_dealloc,
        .tp_init = (initproc) sparse_matrix_init,
        .tp_getset = PySparseMatrix_getsets,
    .tp_members = PySparseMatrix_members,
    .tp_methods = PySparseMatrix_methods
};

PyObject* py_sparse_matrix_from_components(PyObject* data,
    PyObject* indices,
    PyObject* indptr,
    npy_intp nrows,
    npy_intp ncols)
{
    PySparseMatrix* self = (PySparseMatrix*) PySparseMatrix_Type.tp_alloc(
        &PySparseMatrix_Type,
        0);
    if (!self) {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        return NULL;
    }

    Py_SETREF(self->data, data);
    Py_SETREF(self->indices, indices);
    Py_SETREF(self->indptr, indptr);
    self->rows = nrows;
    self->cols = ncols;

    return (PyObject*) self;
}

int init_sparse_matrix(PyObject* module)
{
    if (PyType_Ready(&PySparseMatrix_Type) < 0) { return -1; }

    PyModule_AddObjectRef(module, "SparseMatrix", (PyObject*) &PySparseMatrix_Type);
    return 0;
}




/*
 * Helper functions
 */


enum SMHFlags
{
    SMH_FINAL = 8,
};


static inline void set_indptr(SMHelper* helper, npy_intp index, npy_intp val)
{
    npy_intp* ptr = (npy_intp*) PyArray_GETPTR1(helper->indptr, index);
    *ptr = val;
}



int smh_init(SMHelper* helper,
    PyArray_Descr* dtype,
    npy_intp nrows,
    npy_intp ncols,
    npy_intp nnz_est,
    int format
    )
{
    if (format != SM_CSC && format != SM_CSR) {
        PyErr_SetString(PyExc_ValueError, "Invalid format");
        return -1;
    }

    npy_intp alloc = format == SM_CSC ? ncols : nrows;
    if (nnz_est < 0) {
        nnz_est = alloc;
    }

    void* ptr = PyMem_Malloc(alloc * sizeof(SMHFrame));
    if (ptr == NULL) {
        PyErr_NoMemory();
        return -1;
    }


    npy_intp indptr_size = alloc + 1;
    PyArrayObject* indptr = (PyArrayObject*) PyArray_SimpleNew(1, &indptr_size, NPY_INTP);
    if (indptr == NULL) {
        PyMem_Free(ptr);
        return -1;
    }

    // the creation method steals a reference to dtype, so incref
    Py_INCREF(dtype);
    PyArrayObject* data = (PyArrayObject*) PyArray_SimpleNewFromDescr(1, &nnz_est, dtype);
    if (data == NULL) {
        PyMem_Free(ptr);
        Py_DECREF(indptr);
        return -1;
    }

    PyArrayObject* indices = (PyArrayObject*) PyArray_SimpleNew(1, &nnz_est, NPY_INTP);
    if (indices == NULL) {
        PyMem_Free(ptr);
        Py_DECREF(indptr);
        Py_DECREF(data);
        return -1;
    }

    helper->frames = (SMHFrame*) ptr;
    helper->data = data;
    helper->indices = indices;
    helper->indptr = indptr;
    helper->size = 0;
    helper->alloc = alloc;
    helper->nnz = 0;
    helper->flags = format;
    helper->rows = nrows;
    helper->cols = ncols;

    set_indptr(helper, 0, 0);
    return 0;
}

void smh_free(SMHelper* helper)
{
    Py_XDECREF(helper->data);
    Py_XDECREF(helper->indices);
    Py_XDECREF(helper->indptr);
    PyMem_Free(helper->frames);
}

int smh_insert_frame(SMHelper* helper)
{
    char* data_ptr;
    npy_intp* indices_ptr;
    if (helper->size == helper->alloc) {
        PyErr_SetString(PyExc_RuntimeError,
            "Internal error building sparse matrix; too many frames");
        return -1;
    }

    if (helper->size == 0) {
        // No frames exist yet, add the base pointers
        data_ptr = (char*) PyArray_DATA(helper->data);
        indices_ptr = (npy_intp*) PyArray_DATA(helper->indices);
    } else {
        // Adding a new frame on top existing stack, use previous frame
        // to get data and indices_ptrs
        SMHFrame* prev_frame = smh_current_frame(helper);
        data_ptr = prev_frame->data + prev_frame->size * PyArray_ITEMSIZE(helper->data);
        indices_ptr = prev_frame->indices + prev_frame->size;

        // increase the nnz
        helper->nnz += prev_frame->size;
    }

    // Set the previous size now we have finished working on it.
    set_indptr(helper, helper->size, helper->nnz);

    SMHFrame* new_frame = &helper->frames[helper->size];

    new_frame->data = data_ptr;
    new_frame->indices = indices_ptr;
    new_frame->size = 0;
    ++helper->size;

    return 0;
}

static int smh_resize(SMHelper* helper, npy_intp new_size)
{
    PyArray_Dims newshape = {
        &new_size,
        1
    };

    // resize returns None on success, so we have to decref on success
    // the array is resized inplace.
    PyObject* result = PyArray_Resize(helper->data, &newshape, 0, NPY_CORDER);
    if (result == NULL) {
        return -1;
    }
    Py_DECREF(result);

    result = PyArray_Resize(helper->indices, &newshape, 0, NPY_CORDER);
    if (result == NULL) {
        return -1;
    }
    Py_DECREF(result);

    // We need to adjust all the fames to point to the new data structures
    const npy_intp itemsize = PyArray_ITEMSIZE(helper->data);
    char *data_ptr = PyArray_BYTES(helper->data);
    npy_intp *indices_ptr = PyArray_DATA(helper->indices);
    for (npy_intp i=0; i<helper->size; ++i) {
        SMHFrame* frame = &helper->frames[i];
        frame->data = data_ptr;
        frame->indices = indices_ptr;

        data_ptr += frame->size * itemsize;
        indices_ptr += frame->size;
    }

    return 0;
}

static inline int is_zero(const void* value, int typenum)
{
    switch (typenum) {
        case NPY_FLOAT:
            return *((const float*) value) == 0.0f;
        case NPY_DOUBLE:
            return *((const double*) value) == 0.0;
        case NPY_LONGDOUBLE:
            return *((const long double*) value) == 0.0L;
        // case NPY_HALF:
            // return *((const npy_float16*) value) == 0.0f;
    }
    return 0;
}


static inline void assign(void* dst, const void* src, int typenum)
{
    switch (typenum) {
        case NPY_FLOAT:
            *((float*) dst) = *((float*) src);
            break;
        case NPY_DOUBLE:
            *((double*) dst) = *((double*) src);
            break;
        case NPY_LONGDOUBLE:
            *((long double*) dst) = *((long double*) src);
            break;
        // case NPY_HALF:
            // *((npy_float16*) dst) = *((npy_float16*) src);
            // break;
    }
}

static inline void add_assign(void* dst, const void* src, int typenum)
{
    switch (typenum) {
        case NPY_FLOAT:
            *((float*) dst) += *((float*) src);
            break;
        case NPY_DOUBLE:
            *((double*) dst) += *((double*) src);
            break;
        case NPY_LONGDOUBLE:
            *((long double*) dst) += *((long double*) src);
            break;
        case NPY_HALF:
            *((npy_float16*) dst) += *((npy_float16*) src);
            break;
    }
}


int smh_insert_value_at_index(SMHelper* helper, npy_intp index, const void* value)
{
    const int typenum = PyArray_TYPE(helper->data);

    if (is_zero(value, typenum)) {
        return 0;
    }

    void* new_element_ptr = smh_get_scalar_for_index(helper, index);
    if (new_element_ptr == NULL) {
        return -1;
    }

    add_assign(new_element_ptr, value, typenum);
    return 0;
}

void* smh_get_scalar_for_index(SMHelper* helper, npy_intp index)
{
    SMHFrame* frame = smh_current_frame(helper);
    const npy_intp itemsize = PyArray_ITEMSIZE(helper->data);

    // First find out of we already have the requested index
    npy_intp diff = frame->size;
    npy_intp pos = 0;
    while (diff > 0) {
        npy_intp half = diff / 2;
        npy_intp mid_pos = pos + half;
        npy_intp mid = frame->indices[mid_pos];
        if (index == mid) {
            return frame->data + mid_pos * itemsize;
        }

        if (mid < index) {
            pos = mid_pos + 1;
            diff -= half + 1;
        } else {
            diff = half;
        }
    }

    if (pos < frame->size && frame->indices[pos] == index) {
        return frame->data + pos * itemsize;
    }

    // If we're here, the element was not found and pos holds the position
    // where it should be inserted
    npy_intp curr_maxsize = PyArray_SHAPE(helper->data)[0];
    if (helper->nnz + frame->size == curr_maxsize) {
        // We need to reallocate
        if (smh_resize(helper, curr_maxsize * 2) < 0) {
            // pyexc already set
            return NULL;
        }
    }

    // Shift elements to make room
    if (pos < frame->size) {
        memmove(&frame->indices[pos + 1], &frame->indices[pos],
                (frame->size - pos) * sizeof(npy_intp));
        memmove(frame->data + (pos + 1) * itemsize,
                frame->data + pos * itemsize,
                (frame->size - pos) * itemsize);
    }

    // Insert new element
    frame->indices[pos] = index;
    void* new_element_ptr = frame->data + pos * itemsize;
    insert_zero(new_element_ptr, PyArray_TYPE(helper->data));

    ++frame->size;
    return new_element_ptr;
}


static int smh_finalize(SMHelper* helper, int resize)
{
    if (helper->flags & SMH_FINAL) {
        return 0;
    }

    if (helper->size != helper->alloc) {
        PyErr_SetString(PyExc_RuntimeError,
            "internal error constructing sparse matrix;"
            " final size differs from expected"
            );
        return -1;
    }

    const SMHFrame* last_frame = smh_current_frame(helper);

    // First we need to finalize the nnz count
    helper->nnz += last_frame->size;
    assert(helper->nnz <= PyArray_SHAPE(helper->data)[0]);

    // Add nnz to the end of the indptr array
    set_indptr(helper, helper->size, helper->nnz);

    // trim the data and indices arrays to nnz
    if (resize && smh_resize(helper, helper->nnz) < 0) {
        return -1;
    }

    helper->flags |= SMH_FINAL;
    return 0;
}

PyObject* smh_build_matrix(SMHelper* helper)
{
    // finalize the construction
    if (smh_finalize(helper, 1) < 0) {
        return NULL;
    }

    // Now we allocate the new sparse matrix
    PySparseMatrix* ret = (PySparseMatrix*) PySparseMatrix_Type.tp_alloc(&PySparseMatrix_Type, 0);
    if (ret == NULL) {
        return NULL;
    }

    // move the arrays over to the new spares matrix
    ret->data = (PyObject*) helper->data;
    helper->data = NULL;
    ret->indices = (PyObject*) helper->indices;
    helper->indices = NULL;
    ret->indptr = (PyObject*) helper->indptr;
    helper->indptr = NULL;

    // set the nrows and ncols
    ret->rows = helper->rows;
    ret->cols = helper->cols;
    ret->format = helper->flags & SMH_FLAGS_FORMAT_MASK;

    return (PyObject*) ret;
}
//
// int smh_swap_format(SMHelper* helper)
// {
//     PyArrayObject* new_data = NULL;
//     PyArrayObject* new_indices = NULL;
//     PyArrayObject* new_indptr = NULL;
//     int ret = -1;
//
//     // Make sure the matrix is finalized before we finish
//     // no resize is needed because we reallocate anyway.
//     if (smh_finalize(helper, 0) < 0) {
//         goto finish;
//     }
//
//     npy_intp new_outer_dim = (smh_is_csr(helper) ? helper->cols : helper->rows);
//     npy_intp new_alloc = new_outer_dim + 1;
//
//     new_indptr = (PyArrayObject*) PyArray_SimpleNew(1, &new_alloc, NPY_INTP);
//     if (new_indptr == NULL) {
//         goto finish;
//     }
//
//     new_data = (PyArrayObject*) PyArray_SimpleNew(1, &helper->nnz, PyArray_TYPE(helper->data));
//     if (new_data == NULL) {
//         goto finish;
//     }
//
//     new_indices = (PyArrayObject*) PyArray_SimpleNew(1, &helper->nnz, NPY_INTP);
//     if (new_indices == NULL) {
//         goto finish;
//     }
//
//     void* data_ptr = PyArray_DATA(helper->data);
//     npy_intp* indices_ptr = PyArray_DATA(helper->indices);
//     npy_intp* indptr_ptr = PyArray_DATA(helper->indptr);
//
//     void* new_data_ptr = PyArray_DATA(new_data);
//     npy_intp* new_indices_ptr = PyArray_DATA(new_indices);
//     npy_intp* new_indptr_ptr = PyArray_DATA(new_indptr);
//
//     /*
//      * The procedure here is to walk over the old indices/data
//      */
//     new_indptr_ptr[0] = 0;
//
//
//
//
//
//
//
//     ret = 0;
// finish:
//     Py_XDECREF(new_data);
//     Py_XDECREF(new_indices);
//     Py_XDECREF(new_indptr);
//     return ret;
// }
