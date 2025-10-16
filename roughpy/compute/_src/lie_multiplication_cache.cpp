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

extern "C" {
PyTypeObject PyLieMultiplicationCache_Type = {
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
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION, /* tp_flags */
        nullptr,                                                /* tp_doc */
        nullptr, /* tp_traverse */
        nullptr, /* tp_clear */
        nullptr, /* tp_richcompare */
        0,       /* tp_weaklistoffset */
        nullptr, /* tp_iter */
        nullptr, /* tp_iternext */
        nullptr, /* tp_methods */
        nullptr, /* tp_members */
        nullptr, /* tp_getset */
        nullptr, /* tp_base */
        nullptr, /* tp_dict */
        nullptr, /* tp_descr_get */
        nullptr, /* tp_descr_set */
        0,       /* tp_dictoffset */
        nullptr, /* tp_init */
        nullptr, /* tp_alloc */
        nullptr, /* tp_new */
        nullptr, /* tp_free */
        nullptr, /* tp_is_gc */
};
}

}// namespace

struct PyLieMultiplicationCacheInner {
    std::unordered_map<LieWord, LieCacheEntryPtr, LieWordHash, LieWordEqual>
            cache;
    int32_t width;
};

static PyObject* new_lmc(int32_t width) noexcept
{
    PyLieMultiplicationCache* lmc = reinterpret_cast<PyLieMultiplicationCache*>(
            PyLieMultiplicationCache_Type
                    .tp_alloc(&PyLieMultiplicationCache_Type, 0)
    );

    if (lmc == nullptr) { return reinterpret_cast<PyObject*>(lmc); }

    lmc->inner = static_cast<PyLieMultiplicationCacheInner*>(
            PyMem_Malloc(sizeof(PyLieMultiplicationCacheInner))
    );

    if (lmc->inner == nullptr) {
        Py_DECREF(lmc);
        PyErr_NoMemory();
        return nullptr;
    }

    try {
        // I don't think the constructor can fail, but wrap anyway
        ::new (&lmc->inner->cache)
                std::unordered_set<LieWord, LieWordHash, LieWordEqual>();
    } catch (...) {
        PyMem_Free(lmc->inner);
        Py_DECREF(lmc);
        PyErr_NoMemory();
        return nullptr;
    }

    lmc->inner->width = width;

    return reinterpret_cast<PyObject*>(lmc);
}

PyObject* get_lie_multiplication_cache(PyLieBasis* basis)
{
    PyObject* rpc_internals
            = PyImport_ImportModule("roughpy.compute._rpy_compute_internals");
    if (rpc_internals == nullptr) { return nullptr; }

    PyObject* lmc_cache = PyObject_GetAttrString(rpc_internals, "_lmc_cache");
    Py_DECREF(rpc_internals);
    if (lmc_cache == nullptr) { return nullptr; }

    int32_t width = PyLieBasis_width(basis);
    PyObject* py_width = PyLong_FromLong(width);
    if (py_width == nullptr) { return nullptr; }

    PyObject* cache = PyDict_GetItem(lmc_cache, py_width);
    if (cache == nullptr) {
        cache = new_lmc(width);

        if (cache == nullptr) { goto finish; }

        if (PyDict_SetItem(lmc_cache, py_width, cache) < 0) {
            Py_DECREF(cache);
            goto finish;
        }
    } else {
        Py_INCREF(cache);
    }

finish:
    Py_DECREF(lmc_cache);
    Py_DECREF(py_width);
    return cache;
}

static inline int compute_bracket_half(
        PyLieMultiplicationCache* cache,
        PyLieBasis* basis,
        const LieWord& outer_word,
        LieWord& inner,
        const npy_intp sign,
        std::map<npy_intp, npy_intp>& vals
)
{
    const LieMultiplicationCacheEntry* outer_product
            = PyLieMultiplicationCache_get(cache, basis, &outer_word);
    if (RPY_UNLIKELY(outer_product == nullptr)) {
        // error already set
        return -1;
    }

    for (npy_intp i = 0; i < outer_product->size; ++i) {
        inner.letters[0] = outer_product->data[2 * i];

        const npy_intp val = outer_product->data[2 * i + 1];

        auto* inner_product
                = PyLieMultiplicationCache_get(cache, basis, &inner);
        if (RPY_UNLIKELY(inner_product == nullptr)) {
            // error already set
            return -1;
        }

        for (npy_intp j = 0; j < inner_product->size; ++j) {
            // ReSharper disable once CppTooWideScopeInitStatement
            auto [it, _] = vals.emplace(inner_product->data[2 * j], 0);

            if (RPY_UNLIKELY(
                        (it->second
                         += sign * val * inner_product->data[2 * j + 1])
                        == 0
                )) {
                vals.erase(it);
            }
        }
    }

    return 0;
}

static LieMultiplicationCacheEntry* compute_bracket_slow(
        PyLieMultiplicationCache* cache,
        PyLieBasis* basis,
        const LieWord& word,
        const npy_intp sign
)
{
    LieWord parents, outer_word, inner;

    std::map<npy_intp, npy_intp> vals;

    if (const auto ret
        = PyLieBasis_get_parents(basis, word.letters[1], &parents);
        ret < 0) {
        // error already set
        return nullptr;
    }

    if (parents.letters[0] > 0 && word.letters[0] != parents.letters[0]) {
        outer_word = {
                {word.letters[0], parents.letters[0]}
        };
        inner = {
                {0, parents.letters[1]}
        };

        if (compute_bracket_half(cache, basis, outer_word, inner, sign, vals)
            < 0) {
            return nullptr;
        }
    }

    if (word.letters[0] != parents.letters[1]) {
        outer_word = {
                {word.letters[0], parents.letters[1]}
        };
        inner.letters[1] = parents.letters[0];

        if (compute_bracket_half(cache, basis, outer_word, inner, -sign, vals)
            < 0) {
            return nullptr;
        }
    }

    const npy_intp size = vals.size();
    npy_intp alloc_bytes = sizeof(LieMultiplicationCacheEntry);
    alloc_bytes += 2 * (size - 1) * sizeof(npy_intp);
    auto* entry = static_cast<LieMultiplicationCacheEntry*>(
            PyMem_Malloc(alloc_bytes)
    );

    entry->size = size;
    if (sign == 1) {
        entry->word = word;
    } else {
        entry->word.left = word.letters[1];
        entry->word.right = word.letters[0];
    }

    npy_intp i = 0;
    for (const auto& [key, val] : vals) {
        entry->data[2 * i] = key;
        entry->data[2 * i + 1] = val;
        ++i;
    }

    return entry;
}

static LieMultiplicationCacheEntry* compute_bracket(
        PyLieMultiplicationCache* inner,
        PyLieBasis* basis,
        const LieWord* word,
        int32_t degree
) noexcept
{

    npy_intp sign = 1;
    LieWord target = *word;
    if (target.letters[0] > target.letters[1]) {
        std::swap(target.letters[0], target.letters[1]);
        sign = -1;
    }

    if (const npy_intp pos = PyLieBasis_find_word(basis, &target, degree);
        pos > 0) {
        // the pair belongs to the cache,
        auto* entry = static_cast<LieMultiplicationCacheEntry*>(
                PyMem_Malloc(sizeof(LieMultiplicationCacheEntry))
        );

        if (entry == nullptr) {
            PyErr_NoMemory();
            return nullptr;
        }

        entry->word = *word;
        entry->size = 1;
        entry->data[0] = pos;
        entry->data[1] = sign;

        return entry;
    }

    try {
        return compute_bracket_slow(inner, basis, target, sign);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

const LieMultiplicationCacheEntry* PyLieMultiplicationCache_get(
        PyLieMultiplicationCache* cache,
        PyLieBasis* basis,
        const LieWord* word
)
{
    static constexpr LieMultiplicationCacheEntry empty{{{0, 0}}, 0, {0}};

    // if (!PyLieBasis_Check(basis_ob)) {
    //     PyErr_SetString(PyExc_TypeError, "expected a Lie basis");
    //     return nullptr;
    // }
    // auto* basis = reinterpret_cast<PyLieBasis*>(basis_ob);

    if (PyLieBasis_width(basis) != cache->inner->width) {
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

    const auto lhs_deg = PyLieBasis_degree(basis, word->letters[0]);
    const auto rhs_deg = PyLieBasis_degree(basis, word->letters[1]);
    const auto degree = lhs_deg + rhs_deg;

    if (degree > PyLieBasis_depth(basis)) { return &empty; }

    const auto true_size = PyLieBasis_true_size(basis);
    if (word->letters[0] >= true_size || word->letters[1] >= true_size) {
        return &empty;
    }

    auto& entry = cache->inner->cache[*word];
    if (!entry) { entry.reset(compute_bracket(cache, basis, word, degree)); }

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
    ptr->inner->cache.clear();

    Py_RETURN_NONE;
}

PyLieMultiplicationCacheInner*
lie_multiplication_cache_to_inner(PyObject* cache)
{
    return nullptr;
}

int init_lie_multiplication_cache(PyObject* module)
{
    if (PyType_Ready(&PyLieMultiplicationCache_Type) < 0) { return -1; }

    PyObject* lmc_cache = PyDict_New();
    if (lmc_cache == nullptr) { return -1; }

    if (PyModule_AddObjectRef(module, "_lmc_cache", lmc_cache) < 0) { return -1; }

    return 0;
}

void lmc_dealloc(PyObject* obj)
{
    auto* self = reinterpret_cast<PyLieMultiplicationCache*>(obj);
    self->inner->cache.~unordered_map();
    PyMem_Free(self->inner);
    Py_TYPE(obj)->tp_free(obj);
}

PyObject* lmc_repr(PyObject* obj)
{
    auto* self = reinterpret_cast<PyLieMultiplicationCache*>(obj);
    return PyUnicode_FromFormat(
            "%s(%d)",
            Py_TYPE(obj)->tp_name,
            self->inner->width
    );
}