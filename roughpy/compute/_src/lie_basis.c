#include "lie_basis.h"

#include <stddef.h>
#include <string.h>
#include <structmember.h>

struct _PyLieBasis
{
    PyObject_HEAD
    int32_t width;
    int32_t depth;
    PyObject* degree_begin;
    PyObject* data;
    PyObject* l2t;
    PyObject* t2l;
};


typedef struct _PySparseMatrix
{
    PyObject_HEAD
    PyObject* data;
    PyObject* indices;
    PyObject* indptr;
    npy_intp rows;
    npy_intp cols;
} PySparseMatrix;


/*
 * PyLieBasis Impl
 */


static PyObject* lie_basis_new(
    PyTypeObject* type,
    PyObject* Py_UNUSED(args),
    PyObject* Py_UNUSED(kwargs))
{
    PyLieBasis* self = (PyLieBasis*) type->tp_alloc(type, 0);
    if (!self) { return NULL; }

    Py_XSETREF(self->degree_begin, Py_NewRef(Py_None));
    Py_XSETREF(self->data, Py_NewRef(Py_None));

    Py_XSETREF(self->l2t, PyDict_New());
    Py_XSETREF(self->t2l, PyDict_New());

    return (PyObject*) self;
}

static void lie_basis_dealloc(PyLieBasis* self)
{
    Py_XDECREF(self->degree_begin);
    Py_XDECREF(self->data);
    Py_XDECREF(self->l2t);
    Py_XDECREF(self->t2l);
    Py_TYPE(self)->tp_free((PyObject*) self);
}


static int construct_lie_basis(PyLieBasis* self)
{
    PyObject* data = NULL;
    PyObject* degree_begin = NULL;
    int ret = -1;

    /*
     * For start, let's set the size of the lie basis to
     * the size of the tensor algebra, which is definitely
     * large enough to hold the data. We can resize down
     * later.
     */
    npy_intp alloc_size = 1;
    for (npy_intp i = 1; i <= self->depth; ++i) {
        alloc_size = 1 + alloc_size * self->width;
    }

    npy_intp degree_begin_shape[1] = {self->depth + 2};
    degree_begin = PyArray_SimpleNew(1, degree_begin_shape, NPY_INTP);

    if (degree_begin == NULL) { goto cleanup; }

    npy_intp data_shape[2] = {alloc_size, 2};
    data = PyArray_SimpleNew(2, data_shape, NPY_INTP);

    if (data == NULL) { goto cleanup; }

    npy_intp* data_ptr = (npy_intp*) PyArray_DATA((PyArrayObject*) data);
    npy_intp* db_ptr = (npy_intp*) PyArray_DATA((PyArrayObject*) degree_begin);

    /*
     * Now we build the actual hall set. This is purely
     * computational and doesn't require interaction with
     * any Python objects so we can release the GIL.
     */
    npy_intp size = 1;

    Py_BEGIN_ALLOW_THREADS;
        // Only the "god element" has degree 0
        db_ptr[0] = 0;

        // The "god element" is the first
        data_ptr[0] = 0;
        data_ptr[1] = 0;

        // assign the letters first
        if (self->depth > 0) {

            // letters start at index 1
            db_ptr[1] = 1;

            for (npy_intp letter = 1; letter <= self->width; ++letter) {
                data_ptr[2 * letter] = 0;// data[letter, 0]
                data_ptr[2 * letter + 1] = letter;// data[letter, 1]
            }

            size += self->width;
            db_ptr[2] = size;
        }

        for (npy_intp degree = 2; degree <= self->depth; ++degree) {
            for (npy_intp left_degree = 1; 2 * left_degree <= degree; ++
                 left_degree) {
                npy_intp right_degree = degree - left_degree;
                npy_intp i_lower = db_ptr[left_degree];
                npy_intp i_upper = db_ptr[left_degree + 1];
                npy_intp j_lower = db_ptr[right_degree];
                npy_intp j_upper = db_ptr[right_degree + 1];

                for (npy_intp i = i_lower; i < i_upper; ++i) {
                    npy_intp j_start = (i + 1 > j_lower) ? i + 1 : j_lower;
                    for (npy_intp j = j_start; j < j_upper; ++j) {
                        if (data_ptr[2 * j] <= i) {
                            data_ptr[2 * size] = i;
                            data_ptr[2 * size + 1] = j;
                            ++size;
                        }
                    }
                }

                db_ptr[degree + 1] = size;
            }
        }

    Py_END_ALLOW_THREADS;

    data_shape[0] = size;

    PyObject* resized_data = PyArray_SimpleNew(2, data_shape, NPY_INTP);
    // PyObject* tmp = PyArray_Newshape((PyArrayObject*) data, &dims, NPY_CORDER);
    if (resized_data == NULL) { goto cleanup; }

    npy_intp* dst_ptr = (npy_intp*) PyArray_DATA((PyArrayObject*) resized_data);
    memcpy(dst_ptr, data_ptr, size * sizeof(npy_intp) * 2);

    Py_SETREF(self->data, resized_data);
    // resized_data is now a borrowed reference, clear it to avoid misuse
    resized_data = NULL;
    // Py_XDECREF(self->data);
    // self->data = tmp;
    // tmp = NULL;

    // Move the degree_begin data into the struct;
    // Py_XDECREF(self->degree_begin);
    // self->degree_begin = degree_begin;
    // degree_begin = NULL;
    Py_SETREF(self->degree_begin, degree_begin);
    /*
     * At this point we have transferred owneship of degree_begin to the struct
     * where it rightfully belongs, so the degree_begin variable now does not
     * hold a strong reference. To avoid a use-after-free bug caused by the
     * Py_XDECREF below, we clear this reference.
     */
    degree_begin = NULL;

    ret = 0;

cleanup:
    Py_XDECREF(degree_begin);
    Py_XDECREF(data);

    return ret;
}

static int check_data_and_db(PyLieBasis* self,
                             PyObject* data,
                             PyObject* degree_begin)
{
    if (!PyArray_Check(data)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected numpy array for data argument");
        return -1;
    }

    if (!PyArray_Check(degree_begin)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected numpy array for degree_begin argument");
        return -1;
    }

    PyArrayObject* data_arr = (PyArrayObject*) data;
    PyArrayObject* db_arr = (PyArrayObject*) degree_begin;

    if (PyArray_TYPE(data_arr) != NPY_INTP) {
        PyErr_SetString(PyExc_ValueError,
                        "data must be (pointer-sized) integers");
        return -1;
    }

    if (PyArray_TYPE(db_arr) != NPY_INTP) {
        PyErr_SetString(PyExc_ValueError,
                        "degree_begin must be (pointer-sized) integers");
        return -1;
    }

    if (PyArray_NDIM(data_arr) != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "expected 2-dimensional array for data");
        return -1;
    }

    if (PyArray_DIM(data_arr, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "expected data to of shape (N, 2)");
        return -1;
    }

    if (PyArray_NDIM(db_arr) != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "expected 1-dimensional array for db");
        return -1;
    }

    if (PyArray_DIM(db_arr, 0) < self->depth + 2) {
        PyErr_SetString(PyExc_ValueError,
                        "degree_begin array must be contain at least depth + 2 elements");
        return -1;
    }

    npy_intp size = *(npy_intp*) PyArray_GETPTR1(db_arr, self->depth + 1);

    if (PyArray_DIM(data_arr, 0) < size) {
        PyErr_SetString(PyExc_ValueError,
                        "mismatch in size between data and degree_begin arrays");
        return -1;
    }

    return 0;
}


static int lie_basis_init(PyLieBasis* self, PyObject* args, PyObject* kwargs)
{
    static char* kwords[] = {"width", "depth", "degree_begin", "data", NULL};
    PyObject* degree_begin = NULL;
    PyObject* data = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "ii|OO",
                                     kwords,
                                     &self->width,
                                     &self->depth,
                                     &degree_begin,
                                     &data)) { return -1; }

    if (data == NULL || degree_begin == NULL) {
        if (construct_lie_basis(self) < 0) { return -1; }
    } else {
        // Do some basic sanity checks to make sure
        if (check_data_and_db(self, data, degree_begin) < 0) { return -1; }

        PyObject* tmp = self->data;
        Py_INCREF(data);
        self->data = data;
        Py_XDECREF(tmp);

        tmp = self->degree_begin;
        Py_INCREF(degree_begin);
        self->degree_begin = degree_begin;
        Py_XDECREF(tmp);
    }

    return 0;
}

static PyObject* lie_basis_repr(PyLieBasis* self)
{
    return PyUnicode_FromFormat("LieBasis(%i, %i)",
                                self->width,
                                self->depth);
}


static PyMemberDef PyLieBasis_members[] = {

        {"width", Py_T_INT, offsetof(PyLieBasis, width), READONLY,
         "width of the basis"},
        {"depth", Py_T_INT, offsetof(PyLieBasis, depth), READONLY,
         "depth of the basis"},
        {"degree_begin", Py_T_OBJECT_EX,offsetof(PyLieBasis, degree_begin),
         READONLY, "array of offsets for each degree"},
        {"data", Py_T_OBJECT_EX, offsetof(PyLieBasis, data), 0, "basis data"},
        {NULL}
};


static PyObject* lie_basis_size(PyObject* self, PyObject* Py_UNUSED(arg))
{
    PyLieBasis* self_ = (PyLieBasis*) self;
    if (Py_IsNone(self_->degree_begin)) {
        PyErr_SetString(PyExc_RuntimeError, "degree_begin is None");
        return NULL;
    }
    npy_intp size = *(npy_intp*) PyArray_GETPTR1(
        (PyArrayObject*) self_->degree_begin,
        self_->depth+1);

    return PyLong_FromLong(size - 1);
}

PyMethodDef PyLieBasis_methods[] = {
        {"size", lie_basis_size, METH_NOARGS, "get the size of the Lie basis"},
        {NULL}
};

PyTypeObject PyLieBasis_Type = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = RPY_CPT_TYPE_NAME(LieBasis),
        .tp_basicsize = sizeof(PyLieBasis),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "LieBasis",
        .tp_methods = PyLieBasis_methods,
        .tp_members = PyLieBasis_members,
        .tp_init = (initproc) lie_basis_init,
        .tp_dealloc = (destructor) lie_basis_dealloc,
        .tp_repr = (reprfunc) lie_basis_repr,
        .tp_new = (newfunc) lie_basis_new,
};


/*
 * PySparseMatrix impl
 */


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
};

/**
 * @brief Create a new sparse matrix from the components
 *
 * Steals a reference to each of its arguments.
 */
static PyObject* sparse_matrix_from_components(
    PyObject* data,
    PyObject* indices,
    PyObject* indptr,
    npy_intp nrows,
    npy_intp ncols
)
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

/*
 * External methods
 */


npy_intp lie_basis_size_to_degree(PyLieBasis* lie_basis, int32_t degree)
{
    if (degree <= 0) { return 0; }
    if (degree >= lie_basis->depth + 1) { degree = lie_basis->depth + 1; }

    npy_intp end = *(npy_intp*) PyArray_GETPTR1(
        (PyArrayObject*) lie_basis->degree_begin,
        degree);

    return end - 1;
}

struct L2TFrame
{
    void* data;
    npy_intp* indices;
    npy_intp size;
    npy_intp alloc;
};

struct L2THelper
{
    struct L2TFrame* frames;
    npy_intp size;
    npy_intp alloc;
};

static int init_l2t_helper(struct L2THelper* helper, int32_t size)
{
    void* ptr = PyMem_Malloc(size * sizeof(struct L2TFrame));

    if (ptr == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    helper->frames = (struct L2TFrame*) ptr;
    helper->size = 0;
    helper->alloc = size;
    return 0;
}

static void free_l2t_helper(struct L2THelper* helper)
{
    for (npy_intp i = 0; i < helper->size; ++i) {
        // Py_XDECREF(helper->frames[i].data);
        // Py_XDECREF(helper->frames[i].indices);
        PyMem_Free(helper->frames[i].indices);
        PyMem_Free(helper->frames[i].data);
    }
    PyMem_Free(helper->frames);
}

static void insert_one(void* ptr, int typenum)
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

static void insert_zero(void* ptr, int typenum)
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

static void add_product(void* out,
                        void* left,
                        void* right,
                        int typenum,
                        int sign)
{
#define LB_DO_PRODUCT(TP) \
    (*(TP*) out) += ((TP) sign) * ((*(const TP*) left)*(*(const TP*) right))
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (typenum) {
        case NPY_FLOAT:
            LB_DO_PRODUCT(npy_float);
            break;
        case NPY_DOUBLE:
            LB_DO_PRODUCT(npy_double);
            break;
        case NPY_LONGDOUBLE:
            LB_DO_PRODUCT(npy_longdouble);
    }
#undef LB_DO_PRODUCT
}


static int insert_frame(struct L2THelper* helper,
                        npy_intp size,
                        PyArray_Descr* descr)
{
    void* data_arr = PyMem_Malloc(size * PyDataType_ELSIZE(descr));
    if (data_arr == NULL) { return -1; }

    npy_intp* indices_arr = (npy_intp*) PyMem_Malloc(size * sizeof(npy_intp));
    if (indices_arr == NULL) { PyMem_Free(data_arr); }
    if (indices_arr == NULL) {
        Py_DECREF(data_arr);
        return -1;
    }

    helper->frames[helper->size].data = data_arr;
    helper->frames[helper->size].indices = indices_arr;
    helper->frames[helper->size].size = 0;
    helper->frames[helper->size].alloc = size;
    ++helper->size;

    return 0;
}

static int insert_l2t_letters(struct L2THelper* helper,
                              npy_intp width,
                              PyArray_Descr* descr)
{
    assert(width < helper->alloc);

    for (npy_intp i = 0; i < width; ++i) {
        if (!insert_frame(helper, 1, descr)) { return -1; }

        struct L2TFrame* frame = &helper->frames[helper->size - 1];

        *(npy_intp*) frame->indices = i + 1;
        insert_one(frame->data, descr->type_num);
        ++frame->size;
    }

    return 0;
}

static void* get_scalar_for_index(struct L2TFrame* frame, npy_intp index, PyArray_Descr* descr)
{
    // First find out of we already have the requested index
    npy_intp diff = frame->size;
    npy_intp pos = 0;
    while (diff > 0) {
        npy_intp half = diff / 2;
        npy_intp mid_pos = pos + half;
        npy_intp mid = frame->indices[mid_pos];
        if (index == mid) {
            return frame->data + mid_pos * PyDataType_ELSIZE(descr);
        }

        if (mid < index) {
            pos = mid_pos + 1;
            diff -= half + 1;
        } else {
            diff = half;
        }
    }

    if (frame->indices[pos] == index) {
        return frame->data + pos * PyDataType_ELSIZE(descr);
    }

    // If we're here, the element was not found and pos holds the position
    // where it should be inserted

    if (frame->size == frame->alloc) {
        // We need to reallocate
        void* new_data = PyMem_Realloc(frame->data,
                                       2*(frame->alloc) * PyDataType_ELSIZE(descr));
        if (new_data == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        npy_intp* new_indices = (npy_intp*) PyMem_Realloc(frame->indices,
                                                          2*(frame->alloc) * sizeof(npy_intp));
        if (new_indices == NULL) {
            PyErr_NoMemory();
            PyMem_Free(new_data);
            return NULL;
        }

        frame->data = new_data;
        frame->indices = new_indices;
        frame->alloc *= 2;
    }

    // Shift elements to make room
    if (pos < frame->size) {
        memmove(&frame->indices[pos + 1], &frame->indices[pos],
                (frame->size - pos) * sizeof(npy_intp));
        memmove((char*)frame->data + (pos + 1) * PyDataType_ELSIZE(descr),
                (char*)frame->data + pos * PyDataType_ELSIZE(descr),
                (frame->size - pos) * PyDataType_ELSIZE(descr));
    }

    // Insert new element
    frame->indices[pos] = index;
    void* new_element_ptr = (char*)frame->data + pos * PyDataType_ELSIZE(descr);
    insert_zero(new_element_ptr, descr->type_num);

    ++frame->size;
    return 0;

}


static int insert_l2t_commutator(struct L2THelper* helper,
                                 PyLieBasis* basis,
                                 npy_intp key,
                                 PyArray_Descr* descr) {}


static PyObject* construct_new_l2t(PyLieBasis* basis, PyArray_Descr* dtype)
{
    /*
     * The Lie to Tensor map is represented as a csc matrix with tensor_dim
     * rows and lie_dim columns.
     */

    PyArrayObject* data;
    PyArrayObject* indices;
    PyArrayObject* indptr;

    npy_intp lie_dim = lie_basis_size_to_degree(basis, basis->depth);

    npy_intp tensor_dim = 1;
    for (int32_t d = 0; d <= basis->depth; ++d) {
        tensor_dim = 1 + basis->width * tensor_dim;
    }

    npy_intp shape[2];
    shape[0] = tensor_dim;

    indptr = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_INTP);
    if (indptr == NULL) { return NULL; }

    npy_intp* db_data = PyArray_DATA((PyArrayObject*) basis->degree_begin);
    npy_intp* indptr_data = PyArray_DATA(indptr);

    // The first row contains no data
    indptr_data[0] = 0;
    indptr_data[1] = 0;

    struct L2THelper helper;
    if (!init_l2t_helper(&helper, tensor_dim)) {
        Py_DECREF(indptr);
        return NULL;
    }

    if (!insert_l2t_letters(&helper, basis->width, dtype)) {
        free_l2t_helper(&helper);
        Py_DECREF(indptr);
        return NULL;
    }

    npy_intp begin = db_data[1];
    npy_intp end = db_data[2];

    for (int32_t degree = 2; degree <= basis->depth; ++degree) {
        npy_intp end = db_data[degree + 1];

        npy_intp size = end - begin;

        begin = end;
    }

    return sparse_matrix_from_components(
        data,
        indices,
        indptr,
        tensor_dim,
        lie_dim);

}


PyObject* get_l2t_matrix(PyObject* basis, PyObject* dtype)
{
    if (!PyObject_TypeCheck(basis, &PyLieBasis_Type)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected LieBasis object");
        return NULL;
    }
    PyLieBasis* self = (PyLieBasis*) basis;

    if (!PyObject_TypeCheck(dtype, &PyArrayDescr_Type)) {
        PyErr_SetString(PyExc_TypeError, "expected numpy dtype");
        return NULL;
    }

    PyObject* l2t = PyDict_GetItem(self->l2t, dtype);
    if (l2t) {
        // We already have it, return cached l2t map
        Py_INCREF(l2t);
        return l2t;
    }

    // construct a new map and insert it into the l2t dict

    l2t = construct_new_l2t(self, (PyArray_Descr*) dtype);
    if (l2t == NULL) { return NULL; }

    Py_INCREF(l2t);
    PyDict_SetItem(self->l2t, dtype, l2t);

    return l2t;
}


static PyObject* construct_new_t2l(PyLieBasis* basis, PyArray_Descr* dtype)
{
    return NULL;
}

PyObject* get_t2l_matrix(PyObject* basis, PyObject* dtype)
{
    if (!PyObject_TypeCheck(basis, &PyLieBasis_Type)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected LieBasis object");
        return NULL;
    }
    PyLieBasis* self = (PyLieBasis*) basis;

    if (!PyObject_TypeCheck(dtype, &PyArrayDescr_Type)) {
        PyErr_SetString(PyExc_TypeError, "expected numpy dtype");
        return NULL;
    }

    PyObject* t2l = PyDict_GetItem(self->l2t, dtype);
    if (t2l) {
        // We already have it, return the cached t2l map
        Py_INCREF(t2l);
        return t2l;
    }

    // construct a new map and insert it into the t2l dict
    t2l = construct_new_t2l(self, (PyArray_Descr*) dtype);
    if (t2l == NULL) { return NULL; }

    Py_INCREF(t2l);
    PyDict_SetItem(self->l2t, dtype, t2l);

    return t2l;
}


int init_lie_basis(PyObject* module)
{
    if (PyType_Ready(&PyLieBasis_Type) < 0) { return -1; }
    if (PyType_Ready(&PySparseMatrix_Type) < 0) { return -1; }

    PyModule_AddObjectRef(module, "LieBasis", (PyObject*) &PyLieBasis_Type);

    // In the future we might actually want to expose these to Python,
    // but for now keep it internal
    // PyModule_AddObjectRef(module, "SparseMatrix", (PyObject*) &PySparseMatrix_Type);

    return 0;
}