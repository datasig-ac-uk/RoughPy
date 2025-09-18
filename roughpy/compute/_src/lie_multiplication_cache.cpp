#include "lie_multiplication_cache.h"

#include "lie_basis.h"

#include <unordered_map>
#include <memory>

#include "lie_basis.h"

namespace  {

struct CacheEntryDeleter
{
    void operator()(const LieMultiplicationCacheEntry* entry) const noexcept
    {
        PyMem_Free(const_cast<void*>(static_cast<const void*>(entry)));
    }
};

using LieCacheEntryPtr = std::unique_ptr<LieMultiplicationCacheEntry, CacheEntryDeleter>;


struct LieWordEqual
{
    bool operator()(const LieWord& lhs, const LieWord& rhs) const noexcept
    {
        return lhs.word[0] == rhs.word[0] && lhs.word[1] == rhs.word[1];
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
inline constexpr std::size_t lie_word_hash_prime3 = 0x165667B19E3779F9ULL;
inline constexpr std::size_t lie_word_hash_prime4 = 0x85EBCA77C2B2AE63ULL;
inline constexpr std::size_t lie_word_hash_prime5 = 0x27D4EB2F165667C5ULL;

constexpr std::size_t lie_word_hash_rotate(std::size_t val) noexcept
{
    return (val << 31) | (val >> 33);
}
#else
inline constexpr std::size_t lie_word_hash_prime1 = 0x9E3779B1U;
inline constexpr std::size_t lie_word_hash_prime2 = 0x85EBCA77U;
inline constexpr std::size_t lie_word_hash_prime3 = 0xC2B2AE3DU;
inline constexpr std::size_t lie_word_hash_prime4 = 0x27D4EB2FU;
inline constexpr std::size_t lie_word_hash_prime5 = 0x165667B1U;

constexpr std::size lie_word_hash_rotate(std::size_t val) noexcept
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
        for (npy_intp i=0; i<2; ++i) {
            acc += word.word[i] * lie_word_hash_prime2;
            acc = lie_word_hash_rotate(acc);
            acc *= lie_word_hash_prime1;
        }

        acc += 2;

        return acc;
    }
};


PyTypeObject PyLieMultiplicationCache_Type = {
    .ob_base =PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = RPY_CPT_TYPE_NAME(LieMultiplicationCache),
    .tp_basicsize = sizeof(PyLieMultiplicationCache)
};

} // namespace










struct PyLieMultiplicationCacheInner
{
    std::unordered_map<LieWord, LieCacheEntryPtr, LieWordHash , LieWordEqual> cache;
    int32_t width;
};

PyObject* get_lie_multiplication_cache(PyObject* basis)
{


    return nullptr;
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


    }

    return entry.get();
}

PyObject* lie_mutiplication_cache_clear(PyObject* cache)
{
    return nullptr;
}

PyLieMultiplicationCacheInner* lie_multiplication_cache_to_inner(
    PyObject* cache)
{
    return nullptr;
}

int init_lie_multiplication_cache(PyObject* module)
{


    return 0;
}
