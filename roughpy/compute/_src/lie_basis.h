#ifndef ROUGHPY_COMPUTE__SRC_LIE_BASIS_H
#define ROUGHPY_COMPUTE__SRC_LIE_BASIS_H

#include "py_headers.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * There are two definitions of a Hall set. The common definition is
 * as follows:
 *
 * Let $H$ be a subset of $M(A)$ (the free magma on $A$). The common elements of
 * the definitions from Bourbaki (Lie Groups and Lie Algebras 1-3, II.2.10) and
 * Reutenauer (Chapter 4) are as follows:
 *
 *  1) H has a total order $\leq$.
 *  2) A is contained in H.
 *
 * The definitions then differ slightly. The remaining parts of the definition
 * from Bourbaki are as follows:
 *
 *  3) Every element of $H\cap M^2(A)$ is of the form $(x, y)$ with $x < y$.
 *     (Here $M^2(A)$ denotes the elements of $M(A)$ of length 2.)
 *  4) An element $w\in M(A)$ of length $\geq 3$ belongs to $H$ if and only if
 *     it is of the form $w = (a, (b, c))$ with $a, b, c\in H$, $(b, c) \in H$,
 *     and $b\leq a < (b, c)$ and $b < c$.
 *
 * Bourbaki also places a condition on $\leq$ whereby if $u, v\in H$ with
 * $l(u) < l(v)$ then $u < v$. (Here $l(u)$ denotes the length of $u$.) This
 * condition essentially places the elements of $H$ in degree-order. We impose
 * this condition too, because it is necessary for the degree_begin array to
 * have meaning, though we might change what we mean by "length".
 *
 * The definition from Reutenauer is as follows.
 *
 *  3) For any $h = (h', h'') \in H\setminus A$ one has $h''\in H$ and
 *     $h < h''$.
 *  4) For any tree $w = (h', h'') \in M(A)\setminus A$ one has $w\in H$ if and
 *     only if $h', h''\in H$ and $h' < h''$ and either $h'\in A$ or $h'=(x, y)$
 *     with $h'' \leq y$.
 *
 * The most consequential difference here is the condition is the factor of $w$
 * on which the ordering condition is applied. In Bourbaki, the right factor
 * $(b, c)$ has the condition $b\leq a < (b, c)$. In Reutenauer, it is the
 * left factor $h' = (x, y)$ on which the condition is applied with
 * $h' < h'' \leq y$.
 *
 * Now the standard Hall sets that we construct uses the Bourbaki definition,
 * but one might reasonably expect users to provide Hall sets that follow the
 * Reutenauer definition. To support this, we add some constants that dictate
 * which version of the definition one should use. This will have consequences
 * for how the basis should be used when, for instance, computing brackets,
 * or when checking if basis data is valid.
 *
 * For now, this is only used in the check function.
 */
typedef enum _PyLieBasisMajor
{
    PLB_Major_Bourbaki = 0,
    PLB_Major_Right = PLB_Major_Bourbaki,
    PLB_Major_Reutenauer = 1,
    PLB_Major_Left = PLB_Major_Reutenauer
} PyLieBasisMajor;


typedef enum _PyLieBasisDataOrdering
{
    PLB_UnorderedData = 0,
    PLB_OrderedData = 1
} PyLieBasisDataOrdering;

typedef union _LieWord
{
    npy_intp letters[2];
    struct {
        npy_intp left;
        npy_intp right;
    };
} LieWord;

typedef struct _PyLieBasis PyLieBasis;

extern PyTypeObject PyLieBasis_Type;

/*
 * At the moment, the Hall set we construct obeys the ordering where both
 * left and right keys are kept in lexicographical order. In the future, we
 * might want to allow other bases where this invariant is not satisfied. In
 * that case, we either need a different total ordering on words such that
 * basis->data is in order with respect to this ordering OR fall back to a
 * linear search instead.
 */

// lexicographic order on words
static inline int hall_word_less(const npy_intp* lhs, const npy_intp* rhs)
{
    return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
}

static inline int hall_word_equal(const npy_intp* lhs, const npy_intp* rhs)
{
    return lhs[0] == rhs[0] && lhs[1] == rhs[1];
}

PyObject* get_l2t_matrix(PyObject* basis, PyObject* dtype_obj);

PyObject* get_t2l_matrix(PyObject* basis, PyObject* dtype_obj);

static inline int PyLieBasis_Check(PyObject* obj)
{
    return PyObject_TypeCheck(obj, &PyLieBasis_Type);
}


// flag functions - these will be useful if we move to a flags integer later
RPY_NO_EXPORT
PyLieBasisDataOrdering PyLieBasis_data_ordering(PyLieBasis* basis);

static inline int PyLieBasis_data_is_ordered(PyLieBasis* basis)
{
    return PyLieBasis_data_ordering(basis) == PLB_OrderedData;
}

static inline int PyLieBasis_data_is_unordered(PyLieBasis* basis)
{
    return PyLieBasis_data_ordering(basis) == PLB_UnorderedData;
}

RPY_NO_EXPORT
PyLieBasisMajor PyLieBasis_major(PyLieBasis* basis);

static inline int PyLieBasis_is_left_major(PyLieBasis* basis)
{
    return PyLieBasis_major(basis) == PLB_Major_Left;
}

static inline int PyLieBasis_is_right_major(PyLieBasis* basis)
{
    return PyLieBasis_major(basis) == PLB_Major_Right;
}

// attribute getters
RPY_NO_EXPORT
int32_t PyLieBasis_width(PyLieBasis* basis);

RPY_NO_EXPORT
int32_t PyLieBasis_depth(PyLieBasis* basis);

RPY_NO_EXPORT
npy_intp PyLieBasis_size(PyLieBasis* basis);

RPY_NO_EXPORT
npy_intp PyLieBasis_true_size(PyLieBasis* basis);

RPY_NO_EXPORT
PyArrayObject* PyLieBasis_degree_begin(PyLieBasis* basis);

RPY_NO_EXPORT
PyArrayObject* PyLieBasis_data(PyLieBasis* basis);

/**
 * @brief Finds the position of a target Lie word in a Lie basis, using a
 * specified degree hint or computing it dynamically if the hint is -1.
 *
 * This function performs a binary search to determine the position of a given
 * target word within the specified Lie basis. It supports optimization through
 * a degree hint, which represents the degree of the target word. If the degree
 * hint is not provided (set as -1), the function calculates the degree based on
 * the letters of the target word.
 *
 * If the degree exceeds the depth of the Lie basis, the search terminates early
 * and returns 0.
 *
 * Binary search is performed on the range of basis elements associated with the
 * determined degree, comparing the target with the words in the basis until
 * either an exact match is found or the search fails.
 *
 * @param basis A pointer to the `PyLieBasis` structure that represents the Lie
 * basis for the search.
 * @param target A constant pointer to the `LieWord` structure representing the
 * word to be found.
 * @param degree_hint An integer hint specifying the degree of the target word.
 * If set to -1, the degree is dynamically computed using the degrees of the
 * target letters.
 * @return The position of the target word within the Lie basis if found, or 0
 * if the word is not in the basis. Returns -1 in case of an error (error state
 * will already be set).
 */
RPY_NO_EXPORT
npy_intp PyLieBasis_find_word(
        PyLieBasis* basis,
        const LieWord* target,
        int32_t degree_hint
);

/**
 * @brief Retrieves the parent components of a Lie word at the specified index
 * in a given Lie basis.
 *
 * This function accesses the data stored in the Lie basis and extracts the
 * left and right components (parents) of the word identified by the provided
 * index. The resulting parent components are stored in the given `LieWord`
 * structure. It validates the index to ensure it is within the valid range
 * of the Lie basis size.
 *
 * @param basis A pointer to the `PyLieBasis` structure representing the Lie
 * basis from which the parent components are retrieved.
 * @param index A non-negative integer specifying the index of the word
 * within the basis whose parent components are to be fetched.
 * @param out A pointer to a `LieWord` structure where the parent components
 * (left and right parts) will be stored.
 * @return Returns 0 on successful retrieval of parent components. Returns -1
 * if the index is out of range or if an error occurs (error state will
 * already be set).
 */
RPY_NO_EXPORT
int PyLieBasis_get_parents(PyLieBasis* basis, npy_intp index, LieWord* out);

/**
 * @brief Determines the degree of a specified key within a given Lie basis.
 *
 * This function calculates the degree of a key in a Lie basis by performing a
 * binary search on the degree boundaries stored in the basis. The degree
 * represents the position within the levels of the Lie basis that the key
 * belongs to.
 *
 * The function iterates through the degree boundaries and evaluates the
 * relationship between the key and these boundaries using binary search logic.
 * It returns the degree value corresponding to the identified position.
 *
 * @param basis A pointer to the `PyLieBasis` structure that represents the Lie
 * basis being queried.
 * @param key An integer key whose degree is to be determined within the Lie
 * basis.
 * @return The degree of the key within the Lie basis. If the key does not match
 * any valid degree, the function may return a negative or unexpected value, but
 * behavior outside valid inputs is undefined.
 */
RPY_NO_EXPORT
int32_t PyLieBasis_degree(PyLieBasis* basis, npy_intp key);

PyObject* PyLieBasis_key2str(PyLieBasis* basis, npy_intp key);

PyObject* PyLieBasis_word2str(PyLieBasis* basis, const LieWord* word);

PyObject* PyLieBasis_get(int32_t width, int32_t depth);

/**
 * @brief Expands a target key within a Lie basis into its foliage
 * representation.
 *
 * This function recursively decomposes a given key in the Lie basis into
 * its corresponding foliage, which is the list of letters defining the
 * structure of the key. The decomposition relies on navigating through
 * parent-child relationships in the basis and storing the resulting letters in
 * the provided foliage array.
 *
 * The function uses the provided foliage array as both storage for the
 * result and a temporary stack for internal processing. It ensures that
 * sufficient space is available for both purposes, failing with an error
 * if the maximum size is exceeded or other constraints are violated.
 *
 * @param basis A pointer to the `PyLieBasis` structure representing the Lie
 * basis to operate on.
 * @param key An integer key within the Lie basis to be expanded into its
 * foliage.
 * @param foliage A pointer to an array of integers to store the resulting
 * foliage. The foliage array must be sufficiently large to hold the result and
 * intermediate processing data.
 * @param foliage_maxsize The maximum size of the foliage array. This determines
 * the allowed storage for both the result and any temporary stack data used
 * during processing.
 * @return The number of letters written into the foliage array if the operation
 * is successful. Returns -1 in case of an error, such as insufficient array
 * size or a runtime issue.
 */
RPY_NO_EXPORT
npy_intp PyLieBasis_get_foliage(PyLieBasis* basis, npy_intp key, npy_intp* foliage, npy_intp foliage_maxsize);


RPY_NO_EXPORT
int PyLieBasis_canonicalize_word(PyLieBasis* basis, LieWord* word, int32_t* ldegree, int32_t* rdegree);



int PyLieBasis_check_data_internal(
        PyArrayObject* data,
        PyArrayObject* degree_begin,
        int32_t width,
        int32_t depth,
        PyObject* total_order,
        PyLieBasisMajor major,
        char const** message
);


npy_intp compute_lie_degree_dim(int32_t width, int32_t degree);


npy_intp compute_lie_dim(const int32_t width, const int32_t depth);


PyObject*
PyLieBasis_check_data(PyObject* self, PyObject* args, PyObject* kwargs);

int init_lie_basis(PyObject* module);

#ifdef __cplusplus
}
#endif

#endif// ROUGHPY_COMPUTE__SRC_LIE_BASIS_H
