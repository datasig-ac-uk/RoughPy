#include "lie_multiplication_cache.h"

#include "lie_basis.h"

#include <map>
#include <unordered_map>
#include <memory>




#ifndef RPY_UNLIKELY
#define RPY_UNLIKELY(x) (x)
#endif

namespace {

struct CacheEntryDeleter
{
    void operator()(const LieMultiplicationCacheEntry* entry) const noexcept
    {
        PyMem_Free(const_cast<void*>(static_cast<const void*>(entry)));
    }
};

using LieCacheEntryPtr = std::unique_ptr<LieMultiplicationCacheEntry,
    CacheEntryDeleter>;


struct LieWordEqual
{
    bool operator()(const LieWord& lhs, const LieWord& rhs) const noexcept
    {
        return lhs.letters[0] == rhs.letters[0] && lhs.letters[1] == rhs.letters[1];
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

struct LieWordHash
{
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


PyTypeObject PyLieMultiplicationCache_Type = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = RPY_CPT_TYPE_NAME(LieMultiplicationCache),
        .tp_basicsize = sizeof(PyLieMultiplicationCache)
};

}// namespace


struct PyLieMultiplicationCacheInner
{
    std::unordered_map<LieWord, LieCacheEntryPtr, LieWordHash, LieWordEqual>
    cache;
    int32_t width;
};

PyObject* get_lie_multiplication_cache(PyObject* basis) { return nullptr; }


static LieMultiplicationCacheEntry* compute_bracket_slow(
    PyLieMultiplicationCache* cache, PyLieBasis* basis, const LieWord& word,
    const npy_intp sign
    )
{
    std::map<npy_intp, npy_intp> vals;


    LieWord parents;
    if (const auto ret =  PyLieBasis_get_parents(basis, word.letters[1], &parents); ret < 0) {
        // error already set
        return nullptr;
    }

    LieWord outer_word = {.letters {word.letters[0], parents.letters[0] }};
    auto* outer_product = PyLieMultiplicationCache_get(cache, basis, &outer_word);
    if (RPY_UNLIKELY(outer_product == nullptr)) {
        // error already set
        return nullptr;
    }

    LieWord inner = { .letters = {0, parents.letters[1]}};
    for (npy_intp i=0; i < outer_product->size; ++i) {
        inner.letters[0] = outer_product->data[2*i];

        const npy_intp val = outer_product->data[2*i+1];

        auto* inner_product = PyLieMultiplicationCache_get(cache, basis, &inner);
        if (RPY_UNLIKELY(inner_product == nullptr)) {
            // error already set
            return nullptr;
        }

        for (npy_intp j=0; j<inner_product->size; ++j) {
            // ReSharper disable once CppTooWideScopeInitStatement
            auto [it, _] = vals.emplace(inner_product->data[2*j], 0);

            if (RPY_UNLIKELY((it->second += sign*val * inner_product->data[2*j+1]) == 0)) {
                vals.erase(it);
            }
        }
    }

    outer_word = { .letters = {word.letters[1], parents.letters[1]}};
    outer_product = PyLieMultiplicationCache_get(cache, basis, &outer_word);
    if (RPY_UNLIKELY(outer_product == nullptr)) {
        return nullptr;
    }

    inner.letters[1] = parents.letters[0];
    for (npy_intp i=0; i < outer_product->size; ++i) {
        inner.letters[0] = outer_product->data[2*i];
        const npy_intp val = outer_product->data[2*i+1];

        auto* inner_product = PyLieMultiplicationCache_get(cache, basis, &inner);
        if (RPY_UNLIKELY(inner_product == nullptr)) {
            return nullptr;
        }

        for (npy_intp j=0; j<inner_product->size; ++j) {
            // ReSharper disable once CppTooWideScopeInitStatement
            auto [it, _] = vals.emplace(inner_product->data[2*j], 0);

            if (RPY_UNLIKELY((it->second -= sign*val * inner_product->data[2*j+1]) == 0)) {
                vals.erase(it);
            }
        }
    }


    const npy_intp size = vals.size();
    npy_intp alloc_bytes = sizeof(LieMultiplicationCacheEntry);
    if (size > 0) {
        alloc_bytes += (2*size - 1) * sizeof(npy_intp);
    }
    auto* entry = static_cast<LieMultiplicationCacheEntry*>(PyMem_Malloc(alloc_bytes));

    entry->size = size;
    if (sign == 1) {
        entry->word[0] = word.letters[0];
        entry->word[1] = word.letters[1];
    } else {
        entry->word[0] = word.letters[1];
        entry->word[1] = word.letters[0];
    }

    npy_intp i = 0;
    for (const auto& [key, val] : vals) {
        entry->data[2*i] = key;
        entry->data[2*i+1] = val;
        ++i;
    }

    return entry;
}


static LieMultiplicationCacheEntry* compute_bracket(
    PyLieMultiplicationCache* inner,
    PyLieBasis* basis,
    const LieWord* word
) noexcept
{

    npy_intp sign = 1;
    LieWord target = *word;
    if (target.letters[0] > target.letters[1]) {
        std::swap(target.letters[0], target.letters[1]);
        sign = -1;
    }

    if (const npy_intp pos = PyLieBasis_find_word(basis, &target); pos > 0) {
        // the pair belongs to the cache,
        auto* entry = static_cast<LieMultiplicationCacheEntry*>(PyMem_Malloc(
            sizeof(LieMultiplicationCacheEntry) + sizeof(npy_intp)));
        if (entry == nullptr) {
            PyErr_NoMemory();
            return nullptr;
        }

        entry->word[0] = target.letters[0];
        entry->word[1] = target.letters[1];
        entry->size = 1;
        entry->data[0] = pos;
        // ReSharper disable once CppDFAArrayIndexOutOfBounds
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

const LieMultiplicationCacheEntry*
PyLieMultiplicationCache_get(PyLieMultiplicationCache* cache,
                             PyObject* basis_ob,
                             const LieWord* word)
{

    if (!PyLieBasis_Check(basis_ob)) {
        PyErr_SetString(PyExc_TypeError, "expected a Lie basis");
        return nullptr;
    }
    auto* basis = reinterpret_cast<PyLieBasis*>(basis_ob);

    if (PyLieBasis_width(basis) != cache->inner->width) {
        PyErr_SetString(PyExc_ValueError,
                        "width mismatch between basis and cache");
    }

    auto& entry = cache->inner->cache[*word];
    if (!entry) {
        entry.reset(compute_bracket(cache, basis, word));
    }

    return entry.get();
}

PyObject* lie_multiplication_cache_clear(PyObject* cache)
{
    if (PyObject_TypeCheck(cache, &PyLieMultiplicationCache_Type)) {
        PyErr_SetString(PyExc_TypeError, "cannot be used as a Lie multiplication cache");
        return nullptr;
    }

    auto* ptr = reinterpret_cast<PyLieMultiplicationCache*>(cache);
    ptr->inner->cache.~unordered_map();

    PyMem_Free(ptr->inner);

    Py_RETURN_NONE;
}

PyLieMultiplicationCacheInner* lie_multiplication_cache_to_inner(
    PyObject* cache) { return nullptr; }

int init_lie_multiplication_cache(PyObject* module) { return 0; }