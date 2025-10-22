#include "lie_multiplication_cache.h"

#include "lie_basis.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef RPY_UNLIKELY
#  define RPY_UNLIKELY(x) (x)
#endif

extern "C" {
static void lmc_dealloc(PyObject* obj);

static PyObject* lmc_repr(PyObject* obj);
}

namespace {


PyObject* plm_cache_cache = nullptr;

struct CacheEntryDeleter {
    void operator()(const LieMultiplicationCacheEntry* entry) const noexcept
    {
        PyMem_Free(const_cast<void*>(static_cast<const void*>(entry)));
    }
};

using LieCacheEntryPtr
        = std::unique_ptr<LieMultiplicationCacheEntry, CacheEntryDeleter>;

struct LieWordEqual {
    bool operator()(const LieWord& lhs, const LieWord& rhs) const noexcept
    {
        return lhs.letters[0] == rhs.letters[0]
                && lhs.letters[1] == rhs.letters[1];
    }
};

/*
 * We're going to use the same hashing algorithm as Python tuples which is based
 * on the xxhash defined here:
 * https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
 *
 */

#if SIZEOF_VOID_P == 8
inline constexpr std::size_t lie_word_hash_prime1 = 0x9E3779B185EBCA87ULL;
inline constexpr std::size_t lie_word_hash_prime2 = 0xC2B2AE3D27D4EB4FULL;
// inline constexpr std::size_t lie_word_hash_prime3 = 0x165667B19E3779F9ULL;
// inline constexpr std::size_t lie_word_hash_prime4 = 0x85EBCA77C2B2AE63ULL;
inline constexpr std::size_t lie_word_hash_prime5 = 0x27D4EB2F165667C5ULL;

constexpr std::size_t lie_word_hash_rotate(const std::size_t val) noexcept
{
    return (val << 31) | (val >> 33);
}
#else
inline constexpr std::size_t lie_word_hash_prime1 = 0x9E3779B1U;
inline constexpr std::size_t lie_word_hash_prime2 = 0x85EBCA77U;
// inline constexpr std::size_t lie_word_hash_prime3 = 0xC2B2AE3DU;
// inline constexpr std::size_t lie_word_hash_prime4 = 0x27D4EB2FU;
inline constexpr std::size_t lie_word_hash_prime5 = 0x165667B1U;

constexpr std::size lie_word_hash_rotate(const std::size_t val) noexcept
{
    return (val << 13) | (val >> 19);
}
#endif

struct LieWordHash {
    std::size_t operator()(const LieWord& word) const noexcept
    {
        constexpr std::size_t seed = 0;

        auto acc = seed + lie_word_hash_prime5;
        for (npy_intp i = 0; i < 2; ++i) {
            acc += word.letters[i] * lie_word_hash_prime2;
            acc = lie_word_hash_rotate(acc);
            acc *= lie_word_hash_prime1;
        }

        acc += 2;

        return acc;
    }
};

}// namespace

/* clang-format off */
struct PyLieMultiplicationCache {
    PyObject_HEAD
    std::unordered_map<LieWord, LieCacheEntryPtr, LieWordHash, LieWordEqual>
                    cache;
    int32_t width;
};
/* clang-format on */

static PyTypeObject PyLieMultiplicationCache_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
                RPY_CPT_TYPE_NAME(LieMultiplicationCache), /* tp_name */
        sizeof(PyLieMultiplicationCache),                  /* tp_basicsize */
        0,                                                 /* tp_itemsize */
        reinterpret_cast<destructor>(lmc_dealloc),         /* tp_dealloc */
        0,                                    /* tp_vectorcall_offset */
        nullptr,                              /* tp_getattr */
        nullptr,                              /* tp_setattr */
        nullptr,                              /* tp_as_async */
        reinterpret_cast<reprfunc>(lmc_repr), /* tp_repr */
        nullptr,                              /* tp_as_number */
        nullptr,                              /* tp_as_sequence */
        nullptr,                              /* tp_as_mapping */
        nullptr,                              /* tp_hash */
        nullptr,                              /* tp_call */
        nullptr,                              /* tp_str */
        nullptr,                              /* tp_getattro */
        nullptr,                              /* tp_setattro */
        nullptr,                              /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,                   /* tp_flags */
        nullptr,                              /* tp_doc */
        nullptr,                              /* tp_traverse */
        nullptr,                              /* tp_clear */
        nullptr,                              /* tp_richcompare */
        0,                                    /* tp_weaklistoffset */
        nullptr,                              /* tp_iter */
        nullptr,                              /* tp_iternext */
        nullptr,                              /* tp_methods */
        nullptr,                              /* tp_members */
        nullptr,                              /* tp_getset */
        nullptr,                              /* tp_base */
        nullptr,                              /* tp_dict */
        nullptr,                              /* tp_descr_get */
        nullptr,                              /* tp_descr_set */
        0,                                    /* tp_dictoffset */
        nullptr,                              /* tp_init */
        nullptr,                              /* tp_alloc */
        nullptr,                              /* tp_new */
        nullptr,                              /* tp_free */
        nullptr,                              /* tp_is_gc */
};

PyObject* PyLieMultiplicationCache_new(const int32_t width)
{
    auto* lmc = reinterpret_cast<PyLieMultiplicationCache*>(
            PyLieMultiplicationCache_Type
                    .tp_alloc(&PyLieMultiplicationCache_Type, 0)
    );

    if (lmc == nullptr) { return reinterpret_cast<PyObject*>(lmc); }

    try {
        // I don't think the constructor can fail, but wrap anyway
        ::new (&lmc->cache)
                std::unordered_set<LieWord, LieWordHash, LieWordEqual>();
    } catch (...) {
        Py_DECREF(lmc);
        PyErr_NoMemory();
        return nullptr;
    }

    lmc->width = width;

    return reinterpret_cast<PyObject*>(lmc);
}

PyObject* get_lie_multiplication_cache(PyLieBasis* basis)
{
    // PyObject* rpc_internals
    //         = PyImport_ImportModule("roughpy.compute._rpy_compute_internals");
    // if (rpc_internals == nullptr) { return nullptr; }
    //
    // PyObject* lmc_cache = PyObject_GetAttrString(rpc_internals, "_lmc_cache");
    // Py_DECREF(rpc_internals);
    // if (lmc_cache == nullptr) { return nullptr; }
    if (plm_cache_cache == nullptr) { return nullptr; }

    const int32_t width = PyLieBasis_width(basis);
    PyObject* py_width = PyLong_FromLong(width);
    if (py_width == nullptr) { return nullptr; }

    PyObject* cache = PyDict_GetItem(plm_cache_cache, py_width);
    if (cache == nullptr) {
        cache = PyLieMultiplicationCache_new(width);

        if (cache == nullptr) { goto finish; }

        if (PyDict_SetItem(plm_cache_cache, py_width, cache) < 0) {
            Py_DECREF(cache);
            goto finish;
        }
    } else {
        Py_INCREF(cache);
    }

finish:
    // Py_DECREF(lmc_cache);
    Py_DECREF(py_width);
    return cache;
}

/*
 * Compute a bracket which is either of the form [[x, y], z] or [x, [y, z]], as
 * determined by inner_pos.
 *
 * We first replace the inner bracket (outer_word) by expanding to sum_i a_i u_i
 * where a_i are integers and u_i are basis elements. Then expand by linearity
 * to write
 *
 * [[x, y], z] = [u, z] = sum_i a_i [u_i, z]
 *
 * or
 *
 * [x, [y, z]] = sum a_i [x, u_i].
 *
 * The inner word therefore refers to the brackets of the form [u_i, z] or
 * [x, u_i] that appear in the sum.
 */
static inline int compute_bracket_half(
        PyLieMultiplicationCache* cache,
        PyLieBasis* basis,
        const LieWord& outer_word,
        LieWord& inner_word,
        const npy_intp sign,
        int inner_pos,
        std::map<npy_intp, npy_intp>& vals
)
{
    const LieMultiplicationCacheEntry* outer_product
            = PyLieMultiplicationCache_get(cache, basis, &outer_word, -1);
    if (RPY_UNLIKELY(outer_product == nullptr)) {
        // error already set
        return -1;
    }

    for (npy_intp i = 0; i < outer_product->size; ++i) {
        inner_word.letters[inner_pos] = outer_product->data[2 * i];

        const npy_intp val = outer_product->data[2 * i + 1];

        auto* inner_product
                = PyLieMultiplicationCache_get(cache, basis, &inner_word, -1);
        if (RPY_UNLIKELY(inner_product == nullptr)) {
            // error already set
            return -1;
        }

        for (npy_intp j = 0; j < inner_product->size; ++j) {
            // ReSharper disable once CppTooWideScopeInitStatement
            auto [it, _] = vals.emplace(inner_product->data[2 * j], 0);

            it->second += sign * val * inner_product->data[2 * j + 1];
            if (RPY_UNLIKELY(it->second == 0)) {
                vals.erase(it);
            }
        }
    }

    return 0;
}

static LieCacheEntryPtr new_entry(npy_intp size)
{
    assert(size >= 1);
    LieCacheEntryPtr new_entry(
            static_cast<LieMultiplicationCacheEntry*>(PyMem_Malloc(
                    sizeof(LieMultiplicationCacheEntry)
                    + 2 * (size - 1) * sizeof(npy_intp)
            ))
    );

    if (!new_entry) {
        PyErr_NoMemory();
        return new_entry;
    }

    new_entry->size = size;
    return new_entry;
}


/*
 * If the bracket does not belong to the basis, then we need to use the Jacobi
 * identity to break one of the terms into a Z-linear combination of other
 * brackets and resolve.
 *
 * The jacobi identity says that, for x, y, z, we have
 *
 *  [x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0
 *
 * Combined with antisymmetry, we can write
 *
 *  [x, [y, z]] = [[z, x], y] + [[x, y], z]] = [[x, y], z] - [[x, z], y].
 *
 * Now in our case, we are given a bracket [u, v] to compute, and we should
 * probably expand the factor that has the larger degree. If this is v, then the
 * above is the expansion that is required, but if not then we need to use the
 * antisymmetry once again to obtain
 *
 *  [[y, z], x] = -[x, [y, z]] = [[x, z], y] - [[x, y], z].
 *
 * Of course, it might be good to also reverse the bracket majoring on the right
 * hand side too but regardless, we should get the same answer in the end.
 *
 *
 */
static LieCacheEntryPtr compute_bracket_jacobi(
        PyLieMultiplicationCache* cache,
        PyLieBasis* basis,
        const LieWord& target,
        int32_t lhs_degree,
        int32_t rhs_degree
)
{
    LieWord parents, outer_word, inner_word;

    std::map<npy_intp, npy_intp> vals;

    // This function should only have been called if the end product has degree
    // at least 3.
    assert(lhs_degree > 1 || rhs_degree > 1);

    /*
     * OK this is the complicated part and I'm likely going to forget how this
     * works so it needs some pretty clear documentation. We have to make a
     * choice about which factor of the word to expand using the Jacobi relation
     * to make progress. The decision depends on the majoring of the basis:
     *
     *  - For left-major bases, we expand the left factor if the degree is at
     *    least 2. Otherwise, we expand the right factor.
     *  - For right-major bases, we expand the right factor if the degree is at
     *    least 2. Otherwise, we expand the left factor.
     *
     * The expansion is handled generically by setting two variables
     * `expanded_pos` and `unexpanded_pos` to the values 0 and 1 (or 1 and 0)
     * depending on the basis major and degrees. `expanded_pos` refers to the
     * index in the word corresponding to the factor to be expanded, and
     * `unexpanded_pos = 1 - expanded_pos`. The position into which the
     * unexpanded factor and the parents of the expanded factor are determined
     * by these two integers too. This means we can write some very generic code
     * that uses these integers to put the correct values in the correct places
     * regardless of basis major and which factor is to be expanded.
     *
     * For setting which term to expand, we need to consider which direction the
     * basis itself is majored. For left-major (Reutenaur), we want to expand
     * in the left term provided that lhs_degree > 1. Thus expanded_pos = 0
     * if and only if lhs_degree > 1 but this means that
     * expanded_pos = lhs_degree == 1 which gives 0 if lhs_degree > 1 and 1
     * otherwise. Similarly, for right-major bases we need
     * expanded_pos = rhs_degree > 1 to get expanded_pos = 1 when rhs_degree > 1
     * and 0 otherwise.
     */
    int expanded_pos;
    if (PyLieBasis_is_left_major(basis)) {
        expanded_pos = lhs_degree == 1;
    } else {
        expanded_pos = rhs_degree > 1;
    }
    const int unexpanded_pos = 1-expanded_pos;


    if (PyLieBasis_get_parents(basis, target.letters[expanded_pos], &parents) < 0) {
        // error already set;
        return nullptr;
    }
    assert(parents.left != 0 && parents.right != 0);

    outer_word.letters[unexpanded_pos] = target.letters[unexpanded_pos];

    /*
     * Expand the left factor. This either
     *
     *   [[x, y], z] if the expanded_pos == 1
     *
     * or
     *
     *   [x, [y, z]] if expanded_pos == 0
     *
     * Note that the inner bracket here is the "outer word" since this is
     * computed first, and the outer bracket is the "inner word". The unexpanded
     * position of inner_word is filled in with the terms from resolved outer
     * word expansion.
     */
    if (outer_word.letters[unexpanded_pos] != parents.letters[unexpanded_pos]) {
        outer_word.letters[expanded_pos] = parents.letters[unexpanded_pos];
        inner_word.letters[expanded_pos] = parents.letters[expanded_pos];

        if (compute_bracket_half(cache, basis, outer_word, inner_word, 1, unexpanded_pos, vals)
            < 0) {
            return nullptr;
        }
    }

    /*
     * Expand the right factor. This either
     *
     *   [[x, z], y] if the expanded_pos == 1
     *
     * or
     *
     *   [y, [x, z]] if expanded_pos == 0
     *
     * Note that the inner bracket here is the "outer word" since this is
     * computed first, and the outer bracket is the "inner word". The unexpanded
     * position of inner_word is filled in with the terms from resolved outer
     * word expansion.
     */
    if (outer_word.letters[unexpanded_pos] != parents.letters[expanded_pos]) {
        outer_word.letters[expanded_pos] = parents.letters[expanded_pos];
        inner_word.letters[expanded_pos] = parents.letters[unexpanded_pos];

        if (compute_bracket_half(cache, basis, outer_word, inner_word, -1, unexpanded_pos, vals)
            < 0) {
            return nullptr;
        }
    }

    /*
     * vals now contains the fully expanded brackets with no zeros in it.
     * We need to turn this into a cache entry that we can store and use
     * elsewhere.
     */
    const npy_intp size = vals.size();
    auto entry = new_entry(size);
    entry->word = target;

    npy_intp i = 0;
    for (const auto& [key, val] : vals) {
        entry->data[2 * i] = key;
        entry->data[2 * i + 1] = val;
        ++i;
    }

    return entry;
}

static LieCacheEntryPtr compute_bracket(
        PyLieMultiplicationCache* inner,
        PyLieBasis* basis,
        const LieWord& word,
        const int32_t degree,
        int32_t lhs_degree,
        int32_t rhs_degree
) noexcept
{
    if (const npy_intp pos = PyLieBasis_find_word(basis, &word, degree);
        pos > 0) {
        // the pair belongs to the cache,
        // auto* entry = static_cast<LieMultiplicationCacheEntry*>(
        //         PyMem_Malloc(sizeof(LieMultiplicationCacheEntry))
        // );
        auto entry = new_entry(1);

        if (!entry) {
            // error already set
            return entry;
        }

        entry->word = word;
        entry->data[0] = pos;
        entry->data[1] = 1;

        return entry;
    }

    try {
        return compute_bracket_jacobi(inner, basis, word, lhs_degree, rhs_degree);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

const LieMultiplicationCacheEntry* PyLieMultiplicationCache_get(
        PyLieMultiplicationCache* cache,
        PyLieBasis* basis,
        const LieWord* word,
        int32_t degree
)
{
    static constexpr LieMultiplicationCacheEntry empty{{{0, 0}}, 0, {0}};

    // if (!PyLieBasis_Check(basis_ob)) {
    //     PyErr_SetString(PyExc_TypeError, "expected a Lie basis");
    //     return nullptr;
    // }
    // auto* basis = reinterpret_cast<PyLieBasis*>(basis_ob);

    if (PyLieBasis_width(basis) != cache->width) {
        PyErr_SetString(
                PyExc_ValueError,
                "width mismatch between basis and cache"
        );
    }

    if (word->letters[0] == 0 || word->letters[1] == 0) {
        PyErr_SetString(PyExc_ValueError, "letters must be non-zero");
        return nullptr;
    }

    if (word->letters[0] == word->letters[1]) { return &empty; }

    /*
     * Check if the degree is greater than the basis depth. This is only a
     * screening check because the degree might not actually be given yet, but
     * it does allow us to bypass work in the case where the degree is provided
     * and is obviously too large.
     */
    const auto depth = PyLieBasis_depth(basis);
    if (degree > depth) {
        return &empty;
    }

    /*
     * Another rudimentary check that this is a remotely sensible word to
     * consider computing the product for. If both letters are larger than
     * the basis true size then the degree cannot possibly be less than depth.
     */
    const auto true_size = PyLieBasis_true_size(basis);
    if (word->letters[0] >= true_size || word->letters[1] >= true_size) {
        return &empty;
    }

    /*
     * Look in the cache for word. Either the value is already there, and
     * the returned iterator points to the existing entry, or we insert a new
     * null entry that we will have to populate later; we use the inserted bool
     * to determine which case we're in.
     */
    auto [it, inserted] = cache->cache.emplace(*word, nullptr);
    if (it == cache->cache.end()) {
        PyErr_SetString(PyExc_RuntimeError,
            "could not insert new value into Lie multiplier cache");
        return nullptr;
    }

    // Have we already populated this value in the cache?
    if (!inserted) {
        // We might already be in-flight, but this is an error.
        if (!it->second) {
            PyErr_Format(PyExc_RuntimeError,
                "computing [%zd, %zd] is already in-flight but attempting to access result",
                word->letters[0], word->letters[1]
                );
            return nullptr;
        }
        return it->second.get();
    }

    // local copy of the word so we can modify it
    LieWord target = *word;

    /*
     * If we're here then we haven't found the word in the cache so we need to
     * compute the value. First we need to compute the actual degree and do some
     * more checking that the bracket is worth computing
     */
    auto lhs_deg = PyLieBasis_degree(basis, target.left);
    auto rhs_deg = PyLieBasis_degree(basis, target.right);

    const auto test_degree = lhs_deg + rhs_deg;
    if (degree > 0 && test_degree != degree) {
        PyErr_Format(PyExc_ValueError,
            "mismatched degrees in Lie bracket: %d != %d + %d",
            degree, lhs_deg, rhs_deg
            );
        return nullptr;
    } else {
        // if degree is not already set then set it
        degree = test_degree;
    }

    /*
     * Recheck that degree is not greater than the basis depth. If this is the
     * case, we need to clear out the cache entry before returning the empty
     * product.
     */
    if (degree > depth) {
        cache->cache.erase(it);
        return &empty;
    }

    /*
     * One last thing to check before we start doing actual work. Words have a
     * canonical repesentation according to the basis order. It only makes sense
     * to compute the values for words that have this canonical representation.
     * So our first step is to canonicalize the word. If this involves swapping
     * the letters in the word, then we should first compute the canonical
     * word (possibly by getting it from the cache) and then use this to
     * populate the original cache entry.
     */
    int exchanged = PyLieBasis_canonicalize_word(basis, &target, &lhs_deg, &rhs_deg);
    if (exchanged < 0) {
        return nullptr;
    }

    // Our target is now a canonical word.
    if (!exchanged) {
        // The word was already canonical and doesn't yet exist in the cache, so
        // compute the result and return.
        it->second = compute_bracket(cache, basis, target, degree, lhs_deg, rhs_deg);

        // If it->second is still null then an error occurred and it should be
        // set
        return it->second.get();
    }

    // We had to swap, first look in the cache for the canonical word.
    auto [canon_it, canon_insert] = cache->cache.emplace(target, nullptr);
    if (canon_it == cache->cache.end()) {
        PyErr_SetString(PyExc_RuntimeError,
            "could not insert new entry into Lie multiplier cache");
        return nullptr;
    }
    if (canon_insert) {
        // We inserted a new cache value for the canonicalized word, so we need
        // to populate this.
        canon_it->second = compute_bracket(cache, basis, target, degree, lhs_deg, rhs_deg);

        if (!canon_it->second) {
            // error occurred in computation should be set
            return nullptr;
        }
    }

    /*
     * One way or another we now have a populated canonical entry pointed to
     * by canon_it so now it is time to construct the entry for the original key
     * by just copying the data from the canonical version and reversing the
     * signs of all the scalars.
     */

    auto& entry = it->second;
    const auto& canon = *canon_it->second;
    entry = new_entry(canon.size);
    if (!entry) {
        // error already set
        return nullptr;
    }

    entry->word = *word;
    for (npy_intp i=0; i<canon.size; ++i) {
        entry->data[2 * i] = canon.data[2 * i];
        entry->data[2 * i + 1] = -canon.data[2 * i + 1];
    }


    return entry.get();
}

PyObject* lie_multiplication_cache_clear(PyObject* cache)
{
    if (PyObject_TypeCheck(cache, &PyLieMultiplicationCache_Type)) {
        PyErr_SetString(
                PyExc_TypeError,
                "cannot be used as a Lie multiplication cache"
        );
        return nullptr;
    }

    auto* ptr = reinterpret_cast<PyLieMultiplicationCache*>(cache);
    ptr->cache.clear();

    Py_RETURN_NONE;
}

int init_lie_multiplication_cache(PyObject* module)
{
    if (PyType_Ready(&PyLieMultiplicationCache_Type) < 0) { return -1; }

    // PyObject* lmc_cache = PyDict_New();
    // if (lmc_cache == nullptr) { return -1; }
    plm_cache_cache = PyDict_New();
    // make this immortal?
    if (plm_cache_cache == nullptr) {
        return -1;
    }

    if (PyModule_AddObject(module, "_lmc_cache", plm_cache_cache) < 0) {
        Py_DECREF(plm_cache_cache);
        return -1;
    }

    return 0;
}

void lmc_dealloc(PyObject* obj)
{
    auto* self = reinterpret_cast<PyLieMultiplicationCache*>(obj);
    self->cache.~unordered_map();
    Py_TYPE(obj)->tp_free(obj);
}

PyObject* lmc_repr(PyObject* obj)
{
    const auto* self = reinterpret_cast<PyLieMultiplicationCache*>(obj);
    return PyUnicode_FromFormat("%s(%d)", Py_TYPE(obj)->tp_name, self->width);
}