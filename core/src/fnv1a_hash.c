#include <roughpy/core/fnv1a_hash.h>

/*
 * This is an implementation of the FNV-1A hash function which essentially
 * follows the implementation here: https://github.com/lcn2/fnv. This is the
 * same hash function that is used in NumPy (and is referenced is implemented
 * in the Python source code).
 */


#define FNV1A_MUL_PRIME_64(h) \
    h += (h << 1) + (h << 4) + (h << 5) + (h << 7) + (h << 8) + (h << 40)
#define FNV1A_MUL_PRIME_32(h) \
    h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24)

#define FNV1A_MIX_OCTET_32(h, octet)                                           \
    do {                                                                       \
        h ^= (octet);                                                          \
        FNV1A_MUL_PRIME_32(h);                                                 \
    } while (0)
#define FNV1A_MIX_OCTET_64(h, octet)                                           \
    do {                                                                       \
        h ^= (octet);                                                          \
        FNV1A_MUL_PRIME_64(h);                                                 \
    } while (0)

uint32_t fnv1a_hash_string32(uint32_t state, const char* str)
{
    for (; *str != '\0'; ++str) {
        FNV1A_MIX_OCTET_32(state, *str);
    }
    return state;
}

uint64_t fnv1a_hash_string64(uint64_t state, const char* str)
{
    for (; *str != '\0'; ++str) {
        FNV1A_MIX_OCTET_64(state, *str);
    }
    return state;
}

uint32_t
fnv1a_hash_bytes32(uint32_t state, const void* bytes, const size_t len)
{
    const unsigned char* ptr = (const unsigned char*) bytes;
    const unsigned char* bytes_end = ptr + len;
    for (; ptr != bytes_end; ++ptr) {
        FNV1A_MIX_OCTET_32(state, *ptr);
    }
    return state;
}

uint64_t
fnv1a_hash_bytes64(uint64_t state, const void* bytes, const size_t len)
{
    const unsigned char* ptr = (const unsigned char*) bytes;
    const unsigned char* bytes_end = ptr + len;
    for (; ptr != bytes_end; ++ptr) {
        FNV1A_MIX_OCTET_64(state, *ptr);
    }
    return state;
}
