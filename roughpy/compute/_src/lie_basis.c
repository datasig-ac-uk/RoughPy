#include "lie_basis.h"

#include <stddef.h>
#include <string.h>
#include <structmember.h>

#include "sparse_matrix.h"


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


/*
 * External methods
 */


npy_intp lie_basis_size_to_degree(PyLieBasis* lie_basis, int32_t degree)
{
    if (degree <= 0) { return 0; }
    if (degree >= lie_basis->depth + 1) { degree = lie_basis->depth + 1; }

    npy_intp end = *(npy_intp*) PyArray_GETPTR1(
        (PyArrayObject*) lie_basis->degree_begin,
        degree+1);

    return end - 1;
}

static npy_intp size_of_degree(PyLieBasis* basis, int32_t degree)
{
    if (degree < 1) { return 0; }
    if (degree > basis->depth) { degree = basis->depth; }

    npy_intp being = *(npy_intp*) PyArray_GETPTR1(
        (PyArrayObject*) basis->degree_begin,
        degree);
    npy_intp end = *(npy_intp*) PyArray_GETPTR1(
        (PyArrayObject*) basis->degree_begin,
        degree + 1);

    return end - being;
}

static int32_t degree_of_key(PyLieBasis* basis, npy_intp key)
{
    int32_t diff = basis->depth + 1;
    int32_t pos = 0;
    while (diff > 0) {
        int32_t half = diff / 2;
        int32_t other = pos + half;

        npy_intp* db_ptr = (npy_intp*) PyArray_GETPTR1(
            (PyArrayObject*) basis->degree_begin,
            other);

        if (*db_ptr <= key) {
            pos = other + 1;
            diff -= half + 1;
        } else { diff = half; }
    }

    return pos -1 ;
}

static npy_intp get_tensor_size(PyLieBasis* basis)
{
    npy_intp size = 1;
    for (int32_t i = 1; i <= basis->depth; ++i) {
        size = 1 + basis->width * size;
    }
    return size;
}

static npy_intp get_l2t_nnz_max(PyLieBasis* basis)
{
    npy_intp nnz_est = 0;
    for (int32_t degree = 0; degree < basis->depth; ++degree) {
        npy_intp multiplier = ((npy_intp) 1) << (degree);
        npy_intp count = size_of_degree(basis, degree + 1);

        nnz_est += multiplier * count;
    }
    return nnz_est;
}

static inline void add_product(void* out,
                        void* left,
                        void* right,
                        int typenum,
                        int sign)
{
#define SMH_DO_PRODUCT(TP) \
(*(TP*) out) += ((TP) sign) * ((*(const TP*) left)*(*(const TP*) right))
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (typenum) {
        case NPY_FLOAT:
            SMH_DO_PRODUCT(npy_float);
            break;
        case NPY_DOUBLE:
            SMH_DO_PRODUCT(npy_double);
            break;
        case NPY_LONGDOUBLE:
            SMH_DO_PRODUCT(npy_longdouble);
    }
#undef SMH_DO_PRODUCT
}


static int l2t_insert_letters(SMHelper* helper, npy_intp width)
{
    for (npy_intp i = 0; i < width; ++i) {
        if (smh_insert_frame(helper) < 0) { return -1; }

        void* coeff = smh_get_scalar_for_index(helper, i + 1);
        if (coeff == NULL) {
            return -1;
        }
        insert_one(coeff, smh_dtype(helper));
    }

    return 0;
}

static npy_intp tensor_size_of_degree(PyLieBasis* basis, int32_t degree)
{
    npy_intp size = 1;
    const npy_intp width = basis->width;
    while (degree--) {
        size *= width;
    }
    return size;
}

static int insert_l2t_commutator(SMHelper* helper,
                                 PyLieBasis* basis,
                                 npy_intp key,
                                 int32_t degree
)
{
    const npy_intp itemsize = PyArray_ITEMSIZE(helper->data);

    npy_intp left_key = *(npy_intp*) PyArray_GETPTR2((PyArrayObject*) basis->data, key, 0);
    npy_intp right_key = *(npy_intp*) PyArray_GETPTR2((PyArrayObject*) basis->data, key, 1);

    SMHFrame* left_frame = &helper->frames[left_key - 1];
    SMHFrame* right_frame = &helper->frames[right_key - 1];

    if (smh_insert_frame(helper) < 0) {
        // py exc already set
        return -1;
    }

    const int32_t left_degree = degree_of_key(basis, left_key);
    const int32_t right_degree = degree_of_key(basis, right_key);

    npy_intp left_offset = tensor_size_of_degree(basis, left_degree);
    npy_intp right_offset = tensor_size_of_degree(basis, right_degree);


    for (npy_intp i = 0; i < left_frame->size; ++i) {
        npy_intp left_idx = left_frame->indices[i];
        for (npy_intp j = 0; j < right_frame->size; ++j) {
            npy_intp right_idx = right_frame->indices[j];

            npy_intp idx = left_idx * right_offset + right_idx;
            void* coeff = smh_get_scalar_for_index(helper, idx);
            add_product(coeff,
                        &left_frame->data[i * itemsize],
                        &right_frame->data[j * itemsize],
                        smh_dtype(helper),
                        1);

            idx = right_idx * left_offset + left_idx;
            coeff = smh_get_scalar_for_index(helper, idx);
            add_product(coeff,
                        &right_frame->data[j * itemsize],
                        &left_frame->data[i * itemsize],
                        smh_dtype(helper),
                        -1);
        }
    }

    return 0;
}


static PyObject* construct_new_l2t(PyLieBasis* basis, PyArray_Descr* dtype)
{
    /*
     * The Lie to Tensor map is represented as a csc matrix with tensor_dim
     * rows and lie_dim columns.
     */

    npy_intp lie_dim = lie_basis_size_to_degree(basis, basis->depth);
    npy_intp tensor_dim = get_tensor_size(basis);
    npy_intp nnz_est = get_l2t_nnz_max(basis);

    SMHelper helper;
    if (smh_init(&helper, dtype, lie_dim, nnz_est) < 0) {
        // py exc already set
        return NULL;
    }

    // From now on, if we fail we need to jump to finish to properly clean up
    // so define ret now so it is valid for the return path if we don't
    // finish the build process
    PyObject* ret = NULL;

    if (l2t_insert_letters(&helper, basis->width) < 0) {
        // py exc already set
        goto finish;
    }

    for (int32_t degree = 2; degree <= basis->depth; ++degree) {
        npy_intp key = *(npy_intp*) PyArray_GETPTR1(
            (PyArrayObject*) basis->degree_begin,
            degree);
        npy_intp deg_end = *(npy_intp*) PyArray_GETPTR1(
            (PyArrayObject*) basis->degree_begin,
            degree + 1);

        for (; key < deg_end; ++key) {
            if (insert_l2t_commutator(&helper, basis, key, degree) < 0) {
                // py exc already set
                goto finish;
            }
        }
    }

    ret = smh_build_matrix(&helper, tensor_dim, lie_dim);
finish:
    // whether or not creation was successful, we must free the helper
    smh_free(&helper);
    return ret;
}


PyObject* get_l2t_matrix(PyObject* basis, PyObject* dtype_obj)
{
    if (!PyObject_TypeCheck(basis, &PyLieBasis_Type)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected LieBasis object");
        return NULL;
    }
    PyLieBasis* self = (PyLieBasis*) basis;

    PyArray_Descr* dtype = NULL;
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) {
        return NULL;
    }

    // From this point onwards, we need to make sure we decrement
    // dtype when we're done. We do this with a goto finish statement
    // so set l2t NULL here to make that make sense.
    PyObject* l2t = NULL;

    if (!PyDataType_ISFLOAT(dtype)) {
        PyErr_SetString(PyExc_TypeError,
            "only floating point data types are supported");
        goto finish;
    }

    l2t = PyDict_GetItem(self->l2t, (PyObject*) dtype);
    if (l2t) {
        // We already have it, return cached l2t map
        Py_INCREF(l2t);
        goto finish;
    }

    // construct a new map and insert it into the l2t dict
    // this function does not steal a reference to dtype
    l2t = construct_new_l2t(self, dtype);
    if (l2t == NULL) {
        goto finish;
    }

    // SetItem does not steal a reference to dtype or val.
    PyDict_SetItem(self->l2t, (PyObject*) dtype, l2t);

finish:
    Py_DECREF(dtype);
    return l2t;
}


static PyObject* construct_new_t2l(PyLieBasis* basis, PyArray_Descr* dtype)
{

    Py_RETURN_NOTIMPLEMENTED;
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



PyMethodDef PyLieBasis_methods[] = {
        {"size", lie_basis_size, METH_NOARGS, "get the size of the Lie basis"},
        {"get_l2t_matrix", get_l2t_matrix, METH_O,
         "get a sparse matrix representation of the Lie-to-tensor map"},
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


int init_lie_basis(PyObject* module)
{
    if (PyType_Ready(&PyLieBasis_Type) < 0) { return -1; }

    PyModule_AddObjectRef(module, "LieBasis", (PyObject*) &PyLieBasis_Type);

    return 0;
}