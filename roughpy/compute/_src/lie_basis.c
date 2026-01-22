// ReSharper disable CppParameterMayBeConstPtrOrRef
// ReSharper disable CppLocalVariableMayBeConst
#include "lie_basis.h"

#include <math.h>
#include <stdalign.h>
#include <stddef.h>
#include <string.h>

#define RPY_PYCOMPAT_INCLUDE_STRUCTMEMBER 1
#include <roughpy/pycore/compat.h>

#include <roughpy/pycore/fnv1a_hash.h>

#include "lie_multiplication_cache.h"
#include "sparse_matrix.h"
#include "tensor_basis.h"

struct _PyLieBasis {
    PyObject_HEAD int32_t width;
    int32_t depth;
    PyObject* degree_begin;
    PyObject* data;
    PyObject* l2t;
    PyObject* t2l;
    PyObject* multiplier_cache;

    Py_hash_t cached_hash;
};

// PyLieBasis type functions
static PyObject* lie_basis_new(
        PyTypeObject* type,
        PyObject* _unused_args,
        PyObject* _unused_kwargs
);

static void lie_basis_dealloc(PyLieBasis* self);
static int lie_basis_traverse(PyObject* self, visitproc visit, void* arg);

static int lie_basis_init(PyLieBasis* self, PyObject* args, PyObject* kwargs);

static PyObject* lie_basis_repr(PyLieBasis* self);
static Py_hash_t lie_basis_hash(PyObject* obj);
static PyObject* lie_basis_richcompare(PyObject* self, PyObject* other, int op);

// PylieBasis methods
static PyObject* lie_basis_size(PyObject* self, PyObject* _unused_arg);

static int PyLIeBasis_Major_converter(PyObject* obj, void* result)
{
    const long arg = PyLong_AsLong(obj);

    if (arg == -1 && PyErr_Occurred() != NULL) { return 0; }

    int check = 0;

    check |= arg == PLB_Major_Bourbaki;
    check |= arg == PLB_Major_Reutenauer;

    if (!check) {
        PyErr_Format(
                PyExc_ValueError,
                "Lie basis definition indicator does not match a known "
                "Hall set definition"
        );
        return 0;
    }

    *(PyLieBasis_Major*) result = check;

    return 1;
}

/*******************************************************************************
 * Lie basis type
 ******************************************************************************/

static PyMemberDef PyLieBasis_members[] = {

        {"width",
         Py_T_INT, offsetof(PyLieBasis, width),
         Py_READONLY, "width of the basis"},
        {"depth",
         Py_T_INT, offsetof(PyLieBasis, depth),
         Py_READONLY, "depth of the basis"},
        {"degree_begin",
         Py_T_OBJECT_EX, offsetof(PyLieBasis, degree_begin),
         Py_READONLY, "array of offsets for each degree"},
        {"data", Py_T_OBJECT_EX, offsetof(PyLieBasis, data), 0, "basis data"},
        {NULL}
};

PyMethodDef PyLieBasis_methods[] = {
        {"size", lie_basis_size, METH_NOARGS, "get the size of the Lie basis"},
        {"get_l2t_matrix",
         get_l2t_matrix, METH_O,
         "get a sparse matrix representation of the Lie-to-tensor map"},
        {"get_t2l_matrix",
         (PyCFunction) get_t2l_matrix,
         METH_O, "get the matrix representation of the tensor to Lie map"},
        {NULL}
};

PyTypeObject* PyLieBasis_Type = NULL;

static PyType_Slot lie_basis_slots[] = {
        {        Py_tp_new,         lie_basis_new},
        {       Py_tp_init,        lie_basis_init},
        {    Py_tp_dealloc,     lie_basis_dealloc},
        {   Py_tp_traverse,    lie_basis_traverse},
        {       Py_tp_repr,        lie_basis_repr},
        {       Py_tp_hash,        lie_basis_hash},
        {Py_tp_richcompare, lie_basis_richcompare},
        {    Py_tp_methods,    PyLieBasis_methods},
        {    Py_tp_members,    PyLieBasis_members},
        {        Py_tp_doc,            "LieBasis"},
        {                0,                  NULL}
};

static PyType_Spec lie_basis_spec
        = {.name = "roughpy.LieBasis",
           .basicsize = sizeof(PyLieBasis),
           .itemsize = 0,
           .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
           .slots = lie_basis_slots};

/*******************************************************************************
 * Implementations
 ******************************************************************************/

int init_lie_basis(PyObject* module)
{
    if (init_lie_multiplication_cache(module) < 0) { return -1; }

    PyLieBasis_Type = (PyTypeObject*) PyType_FromSpec(&lie_basis_spec);
    if (PyLieBasis_Type == NULL) { return -1; }

    Py_INCREF(PyLieBasis_Type);
    if (PyModule_AddObject(module, "LieBasis", (PyObject*) PyLieBasis_Type)
        < 0) {
        Py_DECREF(PyLieBasis_Type);
        return -1;
    }

    return 0;
}

static PyObject* lie_basis_new(
        PyTypeObject* type,
        PyObject* Py_UNUSED(args),
        PyObject* Py_UNUSED(kwargs)
)
{
    PyLieBasis* self = (PyLieBasis*) type->tp_alloc(type, 0);
    if (!self) { return NULL; }

    Py_INCREF(Py_None);
    self->degree_begin = Py_None;

    Py_INCREF(Py_None);
    self->data = Py_None;

    self->l2t = PyDict_New();
    self->t2l = PyDict_New();
    self->multiplier_cache = NULL;

    self->cached_hash = -1;

    return (PyObject*) self;
}

static void lie_basis_dealloc(PyLieBasis* self)
{
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->degree_begin);
    Py_XDECREF(self->data);
    Py_XDECREF(self->l2t);
    Py_XDECREF(self->t2l);
    Py_XDECREF(self->multiplier_cache);

    PyTypeObject* type = Py_TYPE(self);
    freefunc tp_free = (freefunc) PyType_GetSlot(type, Py_tp_free);
    if (tp_free) { tp_free(self); }
    Py_DECREF(type);
}

int lie_basis_traverse(PyObject* self, visitproc visit, void* arg)
{
    Py_VISIT(Py_TYPE(self));
    PyLieBasis* lb = (PyLieBasis*) self;

    Py_VISIT(lb->degree_begin);
    Py_VISIT(lb->data);
    Py_VISIT(lb->l2t);
    Py_VISIT(lb->t2l);
    Py_VISIT(lb->multiplier_cache);
    return 0;
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
    // npy_intp alloc_size = 1;
    // for (npy_intp i = 1; i <= self->depth; ++i) {
    //     alloc_size = 1 + alloc_size * self->width;
    // }
    const npy_intp alloc_size = 1 + compute_lie_dim(self->width, self->depth);

    const npy_intp degree_begin_shape[1] = {self->depth + 2};
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
            data_ptr[2 * letter] = 0;         // data[letter, 0]
            data_ptr[2 * letter + 1] = letter;// data[letter, 1]
        }

        size += self->width;
        db_ptr[2] = size;
    }

    for (npy_intp degree = 2; degree <= self->depth; ++degree) {
        for (npy_intp left_degree = 1; 2 * left_degree <= degree;
             ++left_degree) {
            const npy_intp right_degree = degree - left_degree;
            const npy_intp i_lower = db_ptr[left_degree];
            const npy_intp i_upper = db_ptr[left_degree + 1];
            const npy_intp j_lower = db_ptr[right_degree];
            const npy_intp j_upper = db_ptr[right_degree + 1];

            for (npy_intp i = i_lower; i < i_upper; ++i) {
                const npy_intp j_start = (i + 1 > j_lower) ? i + 1 : j_lower;
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

    // data_shape[0] = size;

    // PyObject* resized_data = PyArray_SimpleNew(2, data_shape, NPY_INTP);
    // // PyObject* tmp = PyArray_Newshape((PyArrayObject*) data, &dims,
    // // NPY_CORDER);
    // if (resized_data == NULL) { goto cleanup; }
    //
    // npy_intp* dst_ptr = (npy_intp*) PyArray_DATA((PyArrayObject*)
    // resized_data); memcpy(dst_ptr, data_ptr, size * sizeof(npy_intp) * 2);

    // Py_XSETREF(self->data, resized_data);
    // resized_data is now a borrowed reference, clear it to avoid misuse
    // resized_data = NULL;
    // Py_XDECREF(self->data);
    // self->data = tmp;
    // tmp = NULL;
    Py_XDECREF(self->data);
    self->data = data;
    Py_INCREF(data);
    /*
     * At this point we have transferred ownership of data to the struct where
     * it rightfully belongs, so the data variable now does not hold a strong
     * reference. To avoid a use-after-free bug caused by Py_XDECREF below, we
     * clear this reference.
     */
    data = NULL;

    // Move the degree_begin data into the struct;
    Py_XDECREF(self->degree_begin);
    self->degree_begin = degree_begin;
    Py_INCREF(degree_begin);
    /*
     * At this point we have transferred ownership of degree_begin to the struct
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

#define LBC_SET_ERR(MESSAGE)                                                   \
    do {                                                                       \
        if (msg != NULL) { *msg = MESSAGE; }                                   \
    } while (0)

static int check_data_arrays_basics(
        PyArrayObject* data,
        PyArrayObject* degree_begin,
        int32_t width,
        int32_t depth,
        char const** msg
)
{
    if (!PyArray_IS_C_CONTIGUOUS(degree_begin)) {
        LBC_SET_ERR("degree_begin must be contiguous");
        return 0;
    }

    if (PyArray_NDIM(degree_begin) != 1) {
        LBC_SET_ERR("degree_begin should be 1-dimensional");
        return 0;
    }

    const npy_intp* db_shape = PyArray_SHAPE(degree_begin);
    if (db_shape[0] < depth + 2) {
        LBC_SET_ERR("degree_begin must contain at least depth + 2 entries");
        return 0;
    }

    if (PyArray_TYPE(degree_begin) != NPY_INTP) {
        LBC_SET_ERR("degree_begin must have dtype equal to np.intp");
        return 0;
    }
    const npy_intp* db_data = (npy_intp*) PyArray_DATA(degree_begin);

    if (db_data[0] != 0) {
        LBC_SET_ERR("degree begin must start at 0");
        return 0;
    }
    if (depth >= 1 && db_data[2] - db_data[1] != width) {
        LBC_SET_ERR("mismatch between width and degree_begin for level 1");
        return 0;
    }

    for (int32_t i = 2; i <= depth; ++i) {
        if (db_data[i] <= db_data[i - 1]) {
            LBC_SET_ERR("degree_begin must be strictly increasing");
            return 0;
        }
    }

    if (!PyArray_IS_C_CONTIGUOUS(data)) {
        LBC_SET_ERR("data must be C-contiguous");
        return 0;
    }

    if (PyArray_NDIM(data) != 2) {
        LBC_SET_ERR("data must be 2-dimensional");
        return 0;
    }

    const npy_intp* data_shape = PyArray_SHAPE(data);
    if (data_shape[1] != 2 || data_shape[0] != db_data[depth + 1]) {
        LBC_SET_ERR(
                "data shape must be (N, 2) where N = degree_begin[depth+1]"
        );
        return 0;
    }

    if (PyArray_TYPE(data) != NPY_INTP) {
        LBC_SET_ERR("data must have dtype = np.intp");
        return 0;
    }

    return 1;
}

static int
check_data_and_db(PyLieBasis* self, PyObject* data, PyObject* degree_begin)
{
    if (!PyArray_Check(data)) {
        PyErr_SetString(
                PyExc_TypeError,
                "expected numpy array for data argument"
        );
        return -1;
    }

    if (!PyArray_Check(degree_begin)) {
        PyErr_SetString(
                PyExc_TypeError,
                "expected numpy array for degree_begin argument"
        );
        return -1;
    }

    PyArrayObject* data_arr = (PyArrayObject*) data;
    PyArrayObject* db_arr = (PyArrayObject*) degree_begin;

    const char* msg = NULL;
    if (!check_data_arrays_basics(
                data_arr,
                db_arr,
                self->width,
                self->depth,
                &msg
        )) {
        assert(msg != NULL);
        PyErr_SetString(PyExc_ValueError, msg);
        return -1;
    }

    return 0;
}

static int lie_basis_init(PyLieBasis* self, PyObject* args, PyObject* kwargs)
{
    static char* kwords[] = {"width", "depth", "degree_begin", "data", NULL};
    PyObject* degree_begin = NULL;
    PyObject* data = NULL;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "ii|OO",
                kwords,
                &self->width,
                &self->depth,
                &degree_begin,
                &data
        )) {
        return -1;
    }

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

    PyObject* lmc = get_lie_multiplication_cache(self);
    if (lmc == NULL) {
        PyErr_SetString(
                PyExc_RuntimeError,
                "internal error: failed to get multiplication cache"
        );
        return -1;
    }

    PyObject* tmp = self->multiplier_cache;
    self->multiplier_cache = lmc;
    Py_XDECREF(tmp);

    return 0;
}

static PyObject* lie_basis_repr(PyLieBasis* self)
{ return PyUnicode_FromFormat("LieBasis(%i, %i)", self->width, self->depth); }

static PyObject* lie_basis_size(PyObject* self, PyObject* Py_UNUSED(arg))
{
    const PyLieBasis* self_ = (PyLieBasis*) self;
    if (Py_IsNone(self_->degree_begin)) {
        PyErr_SetString(PyExc_RuntimeError, "degree_begin is None");
        return NULL;
    }
    const npy_intp size = *(
            npy_intp*
    ) PyArray_GETPTR1((PyArrayObject*) self_->degree_begin, self_->depth + 1);

    return PyLong_FromLong(size - 1);
}

Py_hash_t lie_basis_hash(PyObject* obj)
{
    PyLieBasis* self = (PyLieBasis*) obj;

    // This might need to be made atomic in the future
    Py_hash_t hash = self->cached_hash;
    if (hash != -1) { return hash; }

    Py_uhash_t state = FNV1A_OFFSET_BASIS;
    state = fnv1a_hash_bytes(state, "LieBasis", 8);
    state = fnv1a_hash_i32(state, self->width);
    state = fnv1a_hash_i32(state, self->depth);

    /*
     * We need to be a little careful here because the data table might actually
     * contain more values than really belong to the basis. The basis size is
     * thus derived from the PyLieBasis_true_size function, and we read the
     * number of bytes to process from this.
     */
    const size_t data_bytes = 2 * PyLieBasis_true_size(self) * sizeof(npy_intp);
    const void* data = PyArray_DATA((PyArrayObject*) self->data);
    state = fnv1a_hash_bytes(state, data, data_bytes);

    /*
     * we also need to hash the degree_begin array. Again this might actually be
     * larger than the depth requires, so be sure to only process the first
     * depth + 2 values.
     */
    const size_t db_bytes = (self->depth + 2) * sizeof(npy_intp);
    const void* db_data = PyArray_DATA((PyArrayObject*) self->degree_begin);
    state = fnv1a_hash_bytes(state, db_data, db_bytes);

    // There may be other fields we need to hash in here, but not at the moment

    hash = fnv1a_finalize_hash(state);

    // room for atomic store
    self->cached_hash = hash;

    return hash;
}

static inline PyObject*
lie_basis_test_equal(PyLieBasis* left, PyLieBasis* right, long success)
{
    if (left->width != right->width) { return PyBool_FromLong(!success); }

    if (left->depth != right->depth) { return PyBool_FromLong(!success); }

    const npy_intp* l_db_data
            = PyArray_DATA((PyArrayObject*) left->degree_begin);
    const npy_intp* r_db_data
            = PyArray_DATA((PyArrayObject*) right->degree_begin);

    for (int32_t d = 0; d < left->depth + 2; ++d) {
        if (l_db_data[d] != r_db_data[d]) { return PyBool_FromLong(!success); }
    }

    // for now rely on the hash value too. Maybe later we can do something more
    // careful to test that the data is the same.

    Py_hash_t lhash = PyObject_Hash((PyObject*) left);
    Py_hash_t rhash = PyObject_Hash((PyObject*) right);

    if (lhash != rhash) { return PyBool_FromLong(!success); }

    return PyBool_FromLong(success);
}

PyObject* lie_basis_richcompare(PyObject* self, PyObject* other, int op)
{
    if (!PyLieBasis_Check(self) || !PyLieBasis_Check(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    PyLieBasis* left = (PyLieBasis*) self;
    PyLieBasis* right = (PyLieBasis*) other;

    switch (op) {
        case Py_EQ: return lie_basis_test_equal(left, right, 1);
        case Py_NE: return lie_basis_test_equal(left, right, 0);
        case Py_LE:
        case Py_GE:
        case Py_LT:
        case Py_GT:
        default: break;
    }

    Py_RETURN_NOTIMPLEMENTED;
}

/*
 * External methods
 */
int32_t PyLieBasis_width(PyLieBasis* basis) { return basis->width; }

int32_t PyLieBasis_depth(PyLieBasis* basis) { return basis->depth; }

npy_intp PyLieBasis_true_size(PyLieBasis* basis)
{
    return *(
            npy_intp*
    ) PyArray_GETPTR1((PyArrayObject*) basis->degree_begin, basis->depth + 1);
}

npy_intp PyLieBasis_size(PyLieBasis* basis)
{ return PyLieBasis_true_size(basis) - 1; }

PyArrayObject* PyLieBasis_degree_begin(PyLieBasis* basis)
{ return (PyArrayObject*) basis->degree_begin; }

PyArrayObject* PyLieBasis_data(PyLieBasis* basis)
{ return (PyArrayObject*) basis->data; }

static inline void
get_basis_word(PyLieBasis* basis, const npy_intp idx, LieWord* out)
{
    out->letters[0] = *(
            npy_intp*
    ) PyArray_GETPTR2((PyArrayObject*) basis->data, idx, 0);
    out->letters[1] = *(
            npy_intp*
    ) PyArray_GETPTR2((PyArrayObject*) basis->data, idx, 1);
}

int PyLieBasis_get_parents(PyLieBasis* basis, npy_intp index, LieWord* out)
{
    PyArrayObject* data = (PyArrayObject*) basis->data;

    if (index <= 0 || index >= PyLieBasis_true_size(basis)) {
        PyErr_Format(PyExc_IndexError, "index %d out of range");
        return -1;
    }

    out->letters[0] = *(npy_intp*) PyArray_GETPTR2(data, index, 0);

    out->letters[1] = *(npy_intp*) PyArray_GETPTR2(data, index, 1);

    return 0;
}

npy_intp PyLieBasis_find_word(
        PyLieBasis* basis,
        const LieWord* target,
        int32_t degree_hint
)
{

    int32_t degree = degree_hint;
    if (degree == -1) {
        int32_t ldegree = PyLieBasis_degree(basis, target->letters[0]);
        int32_t rdegree = PyLieBasis_degree(basis, target->letters[1]);

        degree = ldegree + rdegree;
    }

    if (degree > basis->depth) { return -1; }

    npy_intp pos = *(
            npy_intp*
    ) PyArray_GETPTR1((PyArrayObject*) basis->degree_begin, degree);
    const npy_intp degree_end = *(
            npy_intp*
    ) PyArray_GETPTR1((PyArrayObject*) basis->degree_begin, degree + 1);

    npy_intp diff = degree_end - pos;

    LieWord test_word;

    // once again, binary search to glory
    while (diff > 0) {
        const npy_intp half = diff / 2;
        const npy_intp other = pos + half;

        get_basis_word(basis, other, &test_word);

        if (hall_word_equal(test_word.letters, target->letters)) {
            return other;
        }

        if (hall_word_less(test_word.letters, target->letters)) {
            pos = other + 1;
            diff -= half + 1;
        } else {
            diff = half;
        }
    }

    get_basis_word(basis, pos, &test_word);
    if (pos < degree_end
        && hall_word_equal(test_word.letters, target->letters)) {
        return pos;
    }

    return -1;
}

int32_t PyLieBasis_degree(PyLieBasis* basis, const npy_intp key)
{
    int32_t diff = basis->depth + 1;
    int32_t pos = 0;
    while (diff > 0) {
        const int32_t half = diff / 2;
        const int32_t other = pos + half;

        const npy_intp* db_ptr = (npy_intp*)
                PyArray_GETPTR1((PyArrayObject*) basis->degree_begin, other);

        if (*db_ptr <= key) {
            pos = other + 1;
            diff -= half + 1;
        } else {
            diff = half;
        }
    }

    return pos - 1;
}

npy_intp lie_basis_size_to_degree(const PyLieBasis* lie_basis, int32_t degree)
{
    if (degree <= 0) { return 0; }
    if (degree >= lie_basis->depth + 1) { degree = lie_basis->depth + 1; }

    const npy_intp end = *(
            npy_intp*
    ) PyArray_GETPTR1((PyArrayObject*) lie_basis->degree_begin, degree + 1);

    return end - 1;
}

static npy_intp size_of_degree(const PyLieBasis* basis, int32_t degree)
{
    if (degree < 1) { return 0; }
    if (degree > basis->depth) { degree = basis->depth; }

    const npy_intp being = *(
            npy_intp*
    ) PyArray_GETPTR1((PyArrayObject*) basis->degree_begin, degree);
    const npy_intp end = *(
            npy_intp*
    ) PyArray_GETPTR1((PyArrayObject*) basis->degree_begin, degree + 1);

    return end - being;
}

static npy_intp get_tensor_size(const PyLieBasis* basis)
{
    npy_intp size = 1;
    for (int32_t i = 1; i <= basis->depth; ++i) {
        size = 1 + basis->width * size;
    }
    return size;
}

static npy_intp get_l2t_nnz_max(const PyLieBasis* basis)
{
    npy_intp nnz_est = 0;
    for (int32_t degree = 0; degree < basis->depth; ++degree) {
        const npy_intp multiplier = ((npy_intp) 1) << (degree);
        const npy_intp count = size_of_degree(basis, degree + 1);

        nnz_est += multiplier * count;
    }
    return nnz_est;
}

static inline void add_product(
        void* out,
        const void* left,
        const void* right,
        const int typenum,
        const int sign
)
{
#define SMH_DO_PRODUCT(TP)                                                     \
    (*(TP*) out) += ((TP) sign) * ((*(const TP*) left) * (*(const TP*) right))
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (typenum) {
        case NPY_FLOAT: SMH_DO_PRODUCT(npy_float); break;
        case NPY_DOUBLE: SMH_DO_PRODUCT(npy_double); break;
        case NPY_LONGDOUBLE: SMH_DO_PRODUCT(npy_longdouble);
    }
#undef SMH_DO_PRODUCT
}

static int l2t_insert_letters(SMHelper* helper, const npy_intp width)
{
    alignas(16) char scratch[16];
    insert_one(scratch, PyArray_TYPE(helper->data));

    for (npy_intp i = 0; i < width; ++i) {
        if (smh_insert_frame(helper) < 0) { return -1; }

        // void* coeff = smh_get_scalar_for_index(helper, i + 1);
        // if (coeff == NULL) { return -1; }
        // insert_one(coeff, smh_dtype(helper));
        smh_insert_value_at_index(helper, i + 1, scratch);
    }

    return 0;
}

static npy_intp tensor_size_of_degree(const PyLieBasis* basis, int32_t degree)
{
    npy_intp size = 1;
    const npy_intp width = basis->width;
    while (degree--) { size *= width; }
    return size;
}

static int insert_l2t_commutator(
        SMHelper* helper,
        PyLieBasis* basis,
        const npy_intp key,
        int32_t degree
)
{
    alignas(16) char scratch[16];

    const npy_intp itemsize = PyArray_ITEMSIZE(helper->data);
    const int typenum = PyArray_TYPE(helper->data);

    const npy_intp left_key = *(
            npy_intp*
    ) PyArray_GETPTR2((PyArrayObject*) basis->data, key, 0);
    const npy_intp right_key = *(
            npy_intp*
    ) PyArray_GETPTR2((PyArrayObject*) basis->data, key, 1);

    const SMHFrame* left_frame = &helper->frames[left_key - 1];
    const SMHFrame* right_frame = &helper->frames[right_key - 1];

    if (smh_insert_frame(helper) < 0) {
        // py exc already set
        return -1;
    }

    const int32_t left_degree = PyLieBasis_degree(basis, left_key);
    const int32_t right_degree = PyLieBasis_degree(basis, right_key);

    const npy_intp left_offset = tensor_size_of_degree(basis, left_degree);
    const npy_intp right_offset = tensor_size_of_degree(basis, right_degree);

    for (npy_intp i = 0; i < left_frame->size; ++i) {
        const npy_intp left_idx = left_frame->indices[i];
        for (npy_intp j = 0; j < right_frame->size; ++j) {
            const npy_intp right_idx = right_frame->indices[j];

            npy_intp idx = left_idx * right_offset + right_idx;
            // void* scratch = smh_get_scalar_for_index(helper, idx);
            insert_zero(scratch, typenum);
            add_product(
                    scratch,
                    &left_frame->data[i * itemsize],
                    &right_frame->data[j * itemsize],
                    typenum,
                    1
            );
            if (smh_insert_value_at_index(helper, idx, scratch) < 0) {
                return -1;
            }

            idx = right_idx * left_offset + left_idx;
            insert_zero(scratch, typenum);
            // coeff = smh_get_scalar_for_index(helper, idx);
            add_product(
                    scratch,
                    &right_frame->data[j * itemsize],
                    &left_frame->data[i * itemsize],
                    typenum,
                    -1
            );
            if (smh_insert_value_at_index(helper, idx, scratch) < 0) {
                return -1;
            }
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

    const npy_intp lie_dim = lie_basis_size_to_degree(basis, basis->depth);
    const npy_intp tensor_dim = get_tensor_size(basis);
    const npy_intp nnz_est = get_l2t_nnz_max(basis);

    SMHelper helper;
    if (smh_init(&helper, dtype, tensor_dim, lie_dim, nnz_est, SM_CSC) < 0) {
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
        npy_intp key = *(
                npy_intp*
        ) PyArray_GETPTR1((PyArrayObject*) basis->degree_begin, degree);
        const npy_intp deg_end = *(
                npy_intp*
        ) PyArray_GETPTR1((PyArrayObject*) basis->degree_begin, degree + 1);

        for (; key < deg_end; ++key) {
            if (insert_l2t_commutator(&helper, basis, key, degree) < 0) {
                // py exc already set
                goto finish;
            }
        }
    }

    ret = smh_build_matrix(&helper);
finish:
    // whether or not creation was successful, we must free the helper
    smh_free(&helper);
    return ret;
}

PyObject* get_l2t_matrix(PyObject* basis, PyObject* dtype_obj)
{
    if (!PyLieBasis_Check(basis)) {
        PyErr_SetString(PyExc_TypeError, "expected LieBasis object");
        return NULL;
    }
    PyLieBasis* self = (PyLieBasis*) basis;

    PyArray_Descr* dtype = NULL;
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) { return NULL; }

    // From this point onwards, we need to make sure we decrement
    // dtype when we're done. We do this with a goto finish statement
    // so set l2t NULL here to make that make sense.
    PyObject* l2t = NULL;

    if (!PyDataType_ISFLOAT(dtype)) {
        PyErr_SetString(
                PyExc_TypeError,
                "only floating point data types are supported"
        );
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
    if (l2t == NULL) { goto finish; }

    // SetItem does not steal a reference to dtype or val.
    PyDict_SetItem(self->l2t, (PyObject*) dtype, l2t);

finish:
    Py_DECREF(dtype);
    return l2t;
}

static int t2l_insert_letters(SMHelper* helper, const npy_intp width)
{
    alignas(16) char scratch[16];
    const int typenum = PyArray_TYPE(helper->data);
    insert_one(scratch, typenum);

    for (npy_intp i = 0; i < width; ++i) {
        if (smh_insert_frame(helper) < 0) { return -1; }

        // void* coeff = smh_get_scalar_for_index(helper, i);
        // if (coeff == NULL) { return -1; }
        // insert_one(coeff, smh_dtype(helper));
        if (smh_insert_value_at_index(helper, i, scratch) < 0) { return -1; }
    }

    return 0;
}

static inline void assign(void* dst, const void* src, const int typenum)
{
    switch (typenum) {
        case NPY_FLOAT: *(npy_float*) dst = *(npy_float*) src; break;
        case NPY_DOUBLE: *(npy_double*) dst = *(npy_double*) src; break;
        case NPY_LONGDOUBLE:
            *(npy_longdouble*) dst = *(npy_longdouble*) src;
            break;
    }
}

static npy_intp get_t2l_nnz_max(PyLieBasis* basis)
{ return get_tensor_size(basis); }

static void one_over_degree(void* scratch, int32_t degree, int typenum)
{
    switch (typenum) {
        case NPY_FLOAT:
            *(npy_float*) scratch = 1.0f / (npy_float) degree;
            break;
        case NPY_DOUBLE:
            *(npy_double*) scratch = 1.0 / (npy_double) degree;
            break;
        case NPY_LONGDOUBLE:
            *(npy_longdouble*) scratch = 1.0 / (npy_longdouble) degree;
            break;
            // case NPY_HALF: do {
            //         float tmp = 1.0f / (float) degree;
            //         *(npy_half*) scratch = npy_half_to_float(tmp);
            //     } while (0);
            //     break;
    }
}

static void multiply_inplace(void* dst, const void* src, const int typenum)
{
    switch (typenum) {
        case NPY_FLOAT: *(npy_float*) dst *= *(npy_float*) src; break;
        case NPY_DOUBLE: *(npy_double*) dst *= *(npy_double*) src; break;
        case NPY_LONGDOUBLE:
            *(npy_longdouble*) dst *= *(npy_longdouble*) src;
            break;
            // case NPY_HALF: do {
            //         float tmp = npy_half_to_float(*(npy_half*) dst);
            //         tmp *= npy_half_to_float(*(npy_half*) src);
            //         *(npy_half*) dst = npy_half_to_float(tmp);
            //     } while (0);
            //     break;
    }
}

// ReSharper disable once CppDFAConstantFunctionResult
static int t2l_normalize_cols(SMHelper* helper, npy_intp width, npy_intp depth)
{
    alignas(16) char scratch[16];

    const int typenum = PyArray_TYPE(helper->data);
    const npy_intp itemsize = PyArray_ITEMSIZE(helper->data);

    npy_intp degree_begin = 1;

    for (int32_t degree = 1; degree <= depth; ++degree) {
        const npy_intp degree_end = 1 + degree_begin * width;

        one_over_degree(scratch, degree, typenum);

        for (npy_intp i = degree_begin; i < degree_end; ++i) {
            SMHFrame* frame = &helper->frames[i];

            // the rbracketing map respects degree, so for each of the columns
            // in our matrix, all the corresponding row entries correspond to
            // lie keys of the same degree as the column index tensor word.
            char* data_bytes = frame->data;
            for (npy_intp j = 0; j < frame->size; ++j) {
                multiply_inplace(&data_bytes[j * itemsize], scratch, typenum);
            }
        }

        degree_begin = degree_end;
    }

    return 0;
}

static int t2l_rbracket(
        PyLieBasis* basis,
        SMHelper* helper,
        const SMHFrame* lframe,
        const SMHFrame* rframe,
        npy_intp first
)
{
    alignas(16) char scratch[16];

    PyLieMultiplicationCache* cache
            = (PyLieMultiplicationCache*) basis->multiplier_cache;

    const npy_intp itemsize = PyArray_ITEMSIZE(helper->data);
    const int typenum = PyArray_TYPE(helper->data);

    LieWord word = {
            {first, 0}
    };
    for (npy_intp i = 0; i < rframe->size; ++i) {
        word.right = 1 + rframe->indices[i];

        const LieMultiplicationCacheEntry* entry
                = PyLieMultiplicationCache_get(cache, basis, &word);
        if (entry == NULL) {
            // py exc already set
            return -1;
        }

        for (npy_intp j = 0; j < entry->size; ++j) {
            npy_intp pkey = entry->data[2 * j];
            npy_intp pval = entry->data[2 * j + 1];

            insert_zero(scratch, typenum);

            add_product(
                    scratch,
                    lframe->data,
                    &rframe->data[i * itemsize],
                    typenum,
                    pval
            );
            if (smh_insert_value_at_index(helper, pkey - 1, scratch) < 0) {
                // py exc already set
                return -1;
            }
        }
    }

    return 0;
}

static PyObject* construct_new_t2l(PyLieBasis* basis, PyArray_Descr* dtype)
{
    /*
     * Tensor to Lie is a sparse linear map represented by a csr matrix. The
     * matrix dimensions are lie_size by tensor_size. The construction is
     * recursive, obtained by repeatedly applying the rbracket map defined
     * by r(1) = 0, r(a) = a for letters a, and r(au) = [a, r(u)] for any
     * letter a and word u. Of course [a, r(u)] need not belong to the basis
     * so there is some non-trivial expansion to be done.
     */

    const npy_intp lie_dim = lie_basis_size_to_degree(basis, basis->depth);
    const npy_intp tensor_dim = get_tensor_size(basis);
    const npy_intp nnz_est = get_t2l_nnz_max(basis);

    SMHelper helper;
    if (smh_init(&helper, dtype, lie_dim, tensor_dim, nnz_est, SM_CSC) < 0) {
        // py exc already set
        return NULL;
    }

    /*
     * From this point on, we must free the helper before we return so define
     * ret now so we can simply goto finish in error states.
     */
    PyObject* ret = NULL;

    // insert the column corresponding to the empty word
    if (smh_insert_frame(&helper) < 0) { goto finish; }

    if (t2l_insert_letters(&helper, basis->width) < 0) {
        // py exc already set
        goto finish;
    }

    npy_intp degm1_begin = 1;
    npy_intp prev_size = basis->width;

    for (int32_t degree = 2; degree <= basis->depth; ++degree) {
        const npy_intp deg_size = prev_size * basis->width;
        const npy_intp deg_begin = 1 + degm1_begin * basis->width;

        for (npy_intp idx = 0; idx < deg_size; ++idx) {
            if (smh_insert_frame(&helper) < 0) {
                // py exc already set
                goto finish;
            }

            // idx / prev_size lies between 0 and width, non-inclusive, but the
            // indices of letters are between 1 and width inclusive.
            const npy_intp lparent = 1 + idx / prev_size;
            // rparent has degree minus 1, so we have to offset
            // idx % prev_size by degm1_begin to get the correct index.
            const npy_intp rparent = degm1_begin + idx % prev_size;

#if (defined(_DEBUG) || defined(RPY_DEBUG) || !defined(NDEBUG))
            if (lparent < 1 || lparent > basis->width) {
                PyErr_SetString(
                        PyExc_RuntimeError,
                        "internal error: lparent is not a letter"
                );
                goto finish;
            }
#endif

            if (lparent == rparent) { continue; }

            // lframe corresponds to a letter, but the 1 is useful to us.
            const SMHFrame* lframe = &helper.frames[lparent];
            const SMHFrame* rframe = &helper.frames[rparent];

            // Compute the bracket [lparent, r(rparent)]
            if (t2l_rbracket(basis, &helper, lframe, rframe, lparent) < 0) {
                // py exc already set
                goto finish;
            }
        }

        // advance the bounds
        degm1_begin = deg_begin;
        prev_size = deg_size;
    }

    /*
     * The rbracket map is only part of the story: r(u) is not the correct image
     * of u in the Lie algebra. To finish off, we must normalize r(u) by
     * 1/deg(u).
     */
    if (t2l_normalize_cols(&helper, basis->width, basis->depth) < 0) {
        // py exc already set
        goto finish;
    }

    // Regardless of whether this call is successful, we still have to free
    // helper.
    ret = smh_build_matrix(&helper);

finish:
    smh_free(&helper);
    return ret;
}

PyObject* get_t2l_matrix(PyObject* basis, PyObject* dtype_obj)
{
    if (!PyLieBasis_Check(basis)) {
        PyErr_SetString(PyExc_TypeError, "expected LieBasis object");
        return NULL;
    }
    PyLieBasis* self = (PyLieBasis*) basis;

    PyArray_Descr* dtype = NULL;
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) { return NULL; }

    PyObject* t2l = PyDict_GetItem(self->t2l, (PyObject*) dtype);
    if (t2l) {
        // We already have it, return the cached t2l map
        Py_INCREF(t2l);
        return t2l;
    }

    // construct a new map and insert it into the t2l dict
    t2l = construct_new_t2l(self, (PyArray_Descr*) dtype);
    if (t2l == NULL) { return NULL; }

    Py_INCREF(t2l);
    PyDict_SetItem(self->l2t, (PyObject*) dtype, t2l);

    return t2l;
}

PyObject* PyLieBasis_key2str(PyLieBasis* basis, const npy_intp key)
{
    if (1 <= key && key <= basis->width) {
        return PyUnicode_FromFormat("%zi", key);
    }

    LieWord parents;
    if (PyLieBasis_get_parents(basis, key, &parents) < 0) { return NULL; }

    return PyLieBasis_word2str(basis, &parents);
}

PyObject* PyLieBasis_word2str(PyLieBasis* basis, const LieWord* word)
{
    PyObject* left_str = PyLieBasis_key2str(basis, word->letters[0]);
    if (left_str == NULL) { return NULL; }

    PyObject* right_str = PyLieBasis_key2str(basis, word->letters[1]);
    if (right_str == NULL) {
        Py_DECREF(left_str);
        return NULL;
    }

    PyObject* ret = PyUnicode_FromFormat("[%U,%U]", left_str, right_str);
    Py_DECREF(left_str);
    Py_DECREF(right_str);
    return ret;
}

PyObject* PyLieBasis_get(int32_t width, int32_t depth)
{
    if (width <= 0 || depth <= 0) {
        PyErr_SetString(
                PyExc_ValueError,
                "width and depth must be positive integers"
        );
        return NULL;
    }

    PyLieBasis* self = (PyLieBasis*) PyType_GenericAlloc(PyLieBasis_Type, 0);
    if (self == NULL) { return NULL; }

    self->width = width;
    self->depth = depth;

    if (construct_lie_basis(self) != 0) {
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject*) self;
}

static int check_lie_data_standard_ordering(
        PyArrayObject* data_arr,
        PyArrayObject* degree_begin_arr,
        int32_t width,
        int32_t depth,
        PyLieBasis_Major major,
        char const** msg
)
{
    if (!check_data_arrays_basics(
                data_arr,
                degree_begin_arr,
                width,
                depth,
                msg
        )) {
        return 0;
    }

    const npy_intp* db_data = (const npy_intp*) PyArray_DATA(degree_begin_arr);
    const npy_intp* data = (const npy_intp*) PyArray_DATA(data_arr);

    if (db_data[1] > 0 && (data[0] != 0 || data[1] != 0)) {
        LBC_SET_ERR("first element of data must be the neutral element");
    }

    // If the depth is 0 then there is nothing more we need to do
    if (depth < 1) { return 1; }

    for (npy_intp l = db_data[1]; l < db_data[2]; ++l) {
        if (data[2 * l + 0] != 0 && data[2 * l + 1] != l) {
            LBC_SET_ERR(
                    "data must contain letters represented as pairs (0, l)"
            );
            return 0;
        }
    }

    for (int32_t degree = 2; degree <= depth; ++degree) {

        for (npy_intp elt_idx = db_data[degree]; elt_idx < db_data[degree + 1];
             ++elt_idx) {

            if (major == PLB_Major_Bourbaki) {

                const npy_intp a = data[2 * elt_idx + 0];
                const npy_intp bc = data[2 * elt_idx + 1];

                if (a >= bc) {
                    LBC_SET_ERR(
                            "elements h = [k,l] in the basis must satisfy k < l"
                    );
                    return 0;
                }

                const npy_intp b = data[2 * bc + 0];
                // const npy_intp c = data[2 * bc + 1];

                if (b != 0 && b > a) {
                    LBC_SET_ERR("when h=[a,[b,c]] we must have b <= a < [b,c]");
                    return 0;
                }
            } else if (major == PLB_Major_Reutenauer) {
                const npy_intp xy = data[2 * elt_idx + 0];
                const npy_intp c = data[2 * elt_idx + 1];

                if (elt_idx >= c) {
                    LBC_SET_ERR("with h=[k,l] we must have h < l");
                    return 0;
                }

                const npy_intp x = data[2 * xy + 0];
                const npy_intp y = data[2 * xy + 1];

                // xy is a letter or xy = [x,y] and y >= c
                if (x != 0 && (x >= y || y < c)) {
                    LBC_SET_ERR(
                            "when h=[[a,b],c] one must have x < y and b >= c"
                    );
                    return 0;
                }
            }
        }
    }

    return 1;
}

typedef int (*order_evaluator)(
        void*,
        npy_intp,
        npy_intp,
        PyArrayObject*,
        PyArrayObject*
);

/**
 * @brief Evaluate the total ordering on two Lie keys
 *
 * Returns true (1) if left <= right and false (0) otherwise. Returns -1 on
 * error, which should be propagated to the Python interpreter.
 */
static int call_py_order_function(
        void* ordering,
        npy_intp left,
        npy_intp right,
        PyArrayObject* data,
        PyArrayObject* db
)
{
    PyObject* result = PyObject_CallFunction(
            (PyObject*) ordering,
            "nnOO",
            left,
            right,
            (PyObject*) data,
            (PyObject*) db
    );

    if (result == NULL) { return -1; }
    // From here on out we need to make sure we decrement result before return

    // For now, just check if the result is truthy
    const int ret = PyObject_IsTrue(result);

    // finish:
    Py_DECREF(result);
    return ret;
}

static int check_lie_data_custom_ordering(
        PyArrayObject* data_arr,
        PyArrayObject* degree_begin_arr,
        int32_t width,
        int32_t depth,
        void* total_order,
        order_evaluator evaluate_order,
        PyLieBasis_Major major,
        char const** msg
)
{
    if (!check_data_arrays_basics(
                data_arr,
                degree_begin_arr,
                width,
                depth,
                msg
        )) {
        return 0;
    }

    const npy_intp* db_data = (const npy_intp*) PyArray_DATA(degree_begin_arr);
    const npy_intp* data = (const npy_intp*) PyArray_DATA(data_arr);

    if (db_data[1] > 0 && (data[0] != 0 || data[1] != 0)) {
        LBC_SET_ERR("first element of data must be the neutral element");
    }

    // If the depth is 0 then there is nothing more we need to do
    if (depth < 1) { return 1; }

    for (npy_intp l = db_data[1]; l < db_data[2]; ++l) {
        if (data[2 * l + 0] != 0 && data[2 * l + 1] != l) {
            LBC_SET_ERR(
                    "data must contain letters represented as pairs (0, l)"
            );
            return 0;
        }
    }

    for (int32_t degree = 2; degree <= depth; ++degree) {

        for (npy_intp elt_idx = db_data[degree]; elt_idx < db_data[degree + 1];
             ++elt_idx) {
            if (major == PLB_Major_Bourbaki) {
                const npy_intp a = data[2 * elt_idx + 0];
                const npy_intp bc = data[2 * elt_idx + 1];

                const int a_le_bc = evaluate_order(
                        total_order,
                        a,
                        bc,
                        data_arr,
                        degree_begin_arr
                );
                if (a_le_bc < 0) {
                    // error already set
                    return -1;
                }

                if (!a_le_bc || a == bc) {
                    LBC_SET_ERR(
                            "elements h = (k, l) in the basis must satisfy k < "
                            "l"
                    );
                    return 0;
                }

                const npy_intp b = data[2 * bc + 0];
                // const npy_intp c = data[2 * bc + 1];

                if (b != 0) {
                    const int b_le_a = evaluate_order(
                            total_order,
                            b,
                            a,
                            data_arr,
                            degree_begin_arr
                    );
                    if (b_le_a < 0) {
                        // error already set
                        return -1;
                    }

                    if (!b_le_a) {
                        LBC_SET_ERR(
                                "when h=[a,[b,c]] we must have b <= a < [b,c]"
                        );
                        return 0;
                    }
                }
            } else if (major == PLB_Major_Reutenauer) {
                const npy_intp xy = data[2 * elt_idx + 0];
                const npy_intp z = data[2 * elt_idx + 1];

                const int h_le_z = evaluate_order(
                        total_order,
                        elt_idx,
                        z,
                        data_arr,
                        degree_begin_arr
                );

                if (!h_le_z || elt_idx == z) {
                    LBC_SET_ERR("when h=[k,l] we must have h < l");
                    return 0;
                }

                const npy_intp x = data[2 * xy + 0];
                const npy_intp y = data[2 * xy + 1];

                if (x != 0) {
                    const int x_le_y = evaluate_order(
                            total_order,
                            x,
                            y,
                            data_arr,
                            degree_begin_arr
                    );

                    if (!x_le_y || x == y) {
                        LBC_SET_ERR("when h=[[a,b],c] we must have x < y");
                        return 0;
                    }

                    const int z_le_y = evaluate_order(
                            total_order,
                            z,
                            y,
                            data_arr,
                            degree_begin_arr
                    );

                    if (!z_le_y) {
                        LBC_SET_ERR("when h=[[a,b],c] we must have b >= c");
                        return 0;
                    }
                }
            }
        }
    }

    return 1;
}

int PyLieBasis_check_data_internal(
        PyArrayObject* data_arr,
        PyArrayObject* degree_begin_arr,
        int32_t width,
        int32_t depth,
        PyObject* total_order,
        PyLieBasis_Major major,
        char const** msg
)
{
    if (total_order == NULL) {
        return check_lie_data_standard_ordering(
                data_arr,
                degree_begin_arr,
                width,
                depth,
                major,
                msg
        );
    }

    if (!PyCallable_Check(total_order)) {
        PyErr_SetString(PyExc_TypeError, "total_order must be callable");
        return -1;
    }

    return check_lie_data_custom_ordering(
            data_arr,
            degree_begin_arr,
            width,
            depth,
            total_order,
            call_py_order_function,
            major,
            msg
    );
}
PyObject* PyLieBasis_check_data(
        PyObject* Py_UNUSED(self),
        PyObject* args,
        PyObject* kwargs
)
{
    static char* kwords[]
            = {"data",
               "degree_begin",
               "width",
               "depth",
               "total_order",
               "basis_major",
               "raise_on_fail"};

    PyObject* ret = NULL;
    PyArrayObject *data = NULL, *degree_begin = NULL;
    PyObject* total_order = NULL;
    int32_t width, depth;
    int throw_on_fail = 0;
    PyLieBasis_Major major = PLB_Major_Bourbaki;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "O&O&ii|OO&p",
                kwords,
                PyArray_Converter,
                &data,
                PyArray_Converter,
                &degree_begin,
                &width,
                &depth,
                &total_order,
                PyLIeBasis_Major_converter,
                &major,
                &throw_on_fail
        )) {
        goto finish;
    }

    const char* message = NULL;
    char const** msg_arg = (throw_on_fail) ? &message : NULL;

    int internal_result = PyLieBasis_check_data_internal(
            data,
            degree_begin,
            width,
            depth,
            total_order,
            major,
            msg_arg
    );

    if (internal_result < 0) { goto finish; }

    if (throw_on_fail && !internal_result) {
        assert(message != NULL);
        PyErr_SetString(PyExc_ValueError, message);
        goto finish;
    }

    ret = PyBool_FromLong(internal_result);

finish:
    Py_XDECREF(data);
    Py_XDECREF(degree_begin);
    return ret;
}

#undef LBC_SET_ERR

/*******************************************************************************
 * Compute degree size
 ******************************************************************************/

/* clang-format off */
static const int8_t mobius_values_cache[32]
        = {0, // dummy
           1, -1, -1,  0, -1,  1, -1,  0,  0,  1,
          -1,  0, -1,  1,  1,  0, -1,  0, -1,  0,
           1,  1, -1,  0,  0,  1,  0,  0, -1, -1,
          -1};
/* clang-format on */

static inline npy_intp mobius_func(int32_t n)
{
    if (n <= 0) { return 0; }

    /*
     * The vast majority of calls to mobius will be for small values. We cache
     * the first 31 such values (+1 implicit value to make the indexing
     * convenient).
     */
    if (n <= 31) { return mobius_values_cache[n]; }

    int n_prime_factors = 0;

    // power 2 can be done efficiently
    if ((n & 1) == 0) {
        n >>= 1;
        if ((n & 1) == 0) {
            // n is not square free, mu(n) = 0
            return 0;
        }
        ++n_prime_factors;
    }

    // test the odd divisors up to sqrt(n)
    // note that the odds that are not primes do waste a division op, but
    // this will always be false since we will have already removed all the
    // prime factors earlier in the process
    for (int32_t d = 3; d * d <= n; d += 2) {
        if ((n % d) == 0) {
            n /= d;
            if ((n % d) == 0) {
                // n is not square free, mu(n) = 0
                return 0;
            }
            ++n_prime_factors;
        }
    }

    // if we haven't already eliminated all the prime factors of n then what
    // remains of n must itself be a prime
    if (n > 1) { ++n_prime_factors; }

    return (n_prime_factors & 1) ? -1 : 1;
}

static inline npy_intp int_pow(npy_intp base, int32_t exp)
{
    npy_intp result = 1;
    npy_intp b = base;
    int32_t e = exp;
    while (e > 0) {
        if (e & 1) { result *= b; }
        e >>= 1;
        if (e) { b *= b; }
    }
    return result;
}

npy_intp compute_lie_degree_dim(int32_t width, int32_t degree)
{
    if (degree <= 0) { return 0; }
    if (degree == 1) { return width; }

    npy_intp sum = 0;
    for (int32_t d = 1; d <= degree; ++d) {
        div_t qr = div(degree, d);
        if (qr.rem == 0) {
            const npy_intp mu = mobius_func(d);
            if (mu != 0) { sum += mu * int_pow(width, qr.quot); }
        }
    }

    return sum / degree;
}
npy_intp compute_lie_dim(const int32_t width, const int32_t depth)
{
    npy_intp result = 0;

    for (int32_t d = 1; d <= depth; ++d) {
        result += compute_lie_degree_dim(width, d);
    }

    return result;
}
