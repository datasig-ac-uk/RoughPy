#ifndef ROUGHPY_SCALARS_SCALAR_TYPE_H_
#define ROUGHPY_SCALARS_SCALAR_TYPE_H_

#include "roughpy_scalars_export.h"
#include "scalars_fwd.h"

#include <functional>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace rpy {
namespace scalars {

struct ScalarTypeInfo {
    BasicScalarInfo basic_info;
    std::string name;
    std::string id;
};

using conversion_function = std::function<void(ScalarPointer, ScalarPointer, dimn_t)>;

namespace dtl {
template <typename T>
struct type_id_of_impl;

}

template <typename T>
inline const std::string &type_id_of() noexcept {
    return dtl::type_id_of_impl<T>::get_id();
}

class ROUGHPY_SCALARS_EXPORT ScalarType {
    ScalarTypeInfo m_info;

protected:
    explicit ScalarType(ScalarTypeInfo info) : m_info(std::move(info)) {}

public:
    template <typename T>
    static const ScalarType *of();

    /*
     * ScalarTypes objects should be unique for each configuration,
     * and should only ever be accessed via a pointer. The deleted
     * copy and move constructors reflect this fact.
     */

    ScalarType(const ScalarType &) = delete;
    ScalarType(ScalarType &&) noexcept = delete;
    ScalarType &operator=(const ScalarType &) = delete;
    ScalarType &operator=(ScalarType &&) noexcept = delete;

    virtual ~ScalarType() = default;

    /**
     * @brief Get the unique internal ID string for this type
     * @return const reference to the ID string.
     */
    const std::string &id() const noexcept { return m_info.id; }

    /**
     * @brief Get the extended scalar type information for this type.
     * @return ScalarTypeInfo struct containing the information.
     */
    const ScalarTypeInfo &info() const noexcept { return m_info; }

    /**
     * @brief Get the size of a single scalar in bytes
     * @return number of bytes.
     */
    int itemsize() const noexcept { return m_info.basic_info.bits / int(sizeof(char)); }

    /**
     * @brief Get the rational type associated with this scalar type
     * @return pointer to the rational type
     */
    const ScalarType* rational_type() const noexcept;

    /**
     * @brief Create a new scalar from numerator and denominator
     * @param numerator Integer numerator of result
     * @param denominator Integer denominator of result
     * @return new Scalar with value numerator/denominator (in the appropriate type)
     */
    virtual Scalar from(long long numerator, long long denominator) const;

    /**
     * @brief Allocate new scalars in memory
     * @param count Number of scalars to allocate space
     * @return ScalarPointer pointing to the newly allocated raw memory.
     */
    virtual ScalarPointer allocate(std::size_t count) const = 0;

    /**
     * @brief Free a previously allocated block of memory
     * @param pointer ScalarPointer pointing to the beginning of the allocated block
     * @param count Number of scalars to be freed.
     */
    virtual void free(ScalarPointer pointer, std::size_t count) const = 0;

    virtual void convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count) const = 0;

    /**
     * @brief Copy count scalars from in to out, converting as necessary
     * @param out raw pointer to output
     * @param in raw pointer to input
     * @param count number of scalars to copy
     * @param info BasicScalarInfo information about the input scalar type
     */
    virtual void convert_copy(void *out, const void *in, std::size_t count, BasicScalarInfo info) const = 0;

    /**
     * @brief Copy count scalars from in to out, converting as necessary
     * @param out raw pointer to destination
     * @param in ScalarPointer to source data
     * @param count number of scalars to copy
     */
    virtual void convert_copy(void *out, ScalarPointer in, std::size_t count) const = 0;

    /**
     * @brief Copy count scalars from in to out, converting as necessary
     * @param out ScalarPointer to destination data
     * @param in raw pointer to source data
     * @param count number of scalars to copy
     * @param id ID of scalar type for source data
     */
    virtual void convert_copy(ScalarPointer out, const void *in, std::size_t count, const std::string &id) const = 0;

    /**
     * @brief
     * @param out
     * @param in
     * @param count
     * @param id
     */
    virtual void convert_fill(ScalarPointer out, ScalarPointer in, dimn_t count, const std::string &id) const;

    /**
     * @brief Get the scalar whose value is one
     * @return new Scalar object
     */
    virtual Scalar one() const;

    /**
     * @brief Get the scalar whose value is minus one
     * @return new Scalar object
     */
    virtual Scalar mone() const;

    /**
     * @brief Get the scalar whose value is zero
     * @return new Scalar object
     */
    virtual Scalar zero() const;

    /**
     * @brief Get the closest scalar_t value to a given scalar
     * @param arg pointer to value
     * @return scalar_t value close to arg in value
     */
    virtual scalar_t to_scalar_t(ScalarPointer arg) const = 0;

    /**
     * @brief assign the rational value numerator/denominator to the target scalar
     * @param target ScalarPointer to old value
     * @param numerator numerator of rational
     * @param denominator denominator of rational
     */
    virtual void assign(ScalarPointer target, long long numerator, long long denominator) const = 0;

    /**
     * @brief Create a copy of a scalar value
     * @param source ScalarPointer to source value
     * @return new Scalar object
     */
    virtual Scalar copy(ScalarPointer source) const;

    /**
     * @brief Get the scalar whose value is minus given value
     * @param arg ScalarPointer to source value
     * @return new Scalar whose value is -(*arg)
     */
    virtual Scalar uminus(ScalarPointer arg) const = 0;

    /**
     * @brief Add one scalar value to another
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     * @return new Scalar with sum of the two values
     */
    virtual Scalar add(ScalarPointer lhs, ScalarPointer rhs) const;

    /**
     * @brief Subract one scalar value to another
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     * @return new Scalar with difference of the two values
     */
    virtual Scalar sub(ScalarPointer lhs, ScalarPointer rhs) const;

    /**
     * @brief Multiply two scalar values
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     * @return new Scalar with product of the two values
     */
    virtual Scalar mul(ScalarPointer lhs, ScalarPointer rhs) const;

    /**
     * @brief Divide one scalar value by another
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     * @return new scalar with left value divided by right value
     */
    virtual Scalar div(ScalarPointer lhs, ScalarPointer rhs) const;

    /**
     * @brief Add right value to left inplcae
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     */
    virtual void add_inplace(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    /**
     * @brief Subract right value from left value inplace
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     */
    virtual void sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    /**
     * @brief Multiply left value by right value inplace
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     */
    virtual void mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    /**
     * @brief Divide left value by right value inplace
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     */
    virtual void div_inplace(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    /**
     * @brief Test if the given value is equal to zero
     * @param arg argument to test
     * @return bool, true if the scalar pointer points to zero
     */
    virtual bool is_zero(ScalarPointer arg) const;

    /**
     * @brief Test if two scalars are equal
     * @param lhs pointer to left value
     * @param rhs pointer to right value
     * @return bool, true if left == right
     */
    virtual bool are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept = 0;

    /**
     * @brief Print the value of a ScalarPointer to an output stream
     * @param arg ScalarPointer to value
     * @param os output stream to print
     */
    virtual void print(ScalarPointer arg, std::ostream &os) const;
};

/**
 * @brief Register a new type with the scalar type system
 * @param type Pointer to newly created ScalarType
 *
 *
 */
ROUGHPY_SCALARS_EXPORT
void register_type(const ScalarType *type);

/**
 * @brief Get a type registered with the scalar type system
 * @param id Id string of type to be retrieved
 * @return pointer to ScalarType representing id
 */
ROUGHPY_SCALARS_EXPORT
const ScalarType *get_type(const std::string &id);

/**
 * @brief Get a list of all registered ScalarTypes
 * @return vector of ScalarType pointers.
 */
ROUGHPY_SCALARS_EXPORT
std::vector<const ScalarType *> list_types();


ROUGHPY_SCALARS_EXPORT
const conversion_function&
get_conversion(const std::string& src_id, const std::string& dst_id);

ROUGHPY_SCALARS_EXPORT
void register_conversion(const std::string& src_id,
                         const std::string& dst_id,
                         conversion_function converter);


inline bool operator==(const ScalarType &lhs, const ScalarType &rhs) noexcept {
    return std::addressof(lhs) == std::addressof(rhs);
}
inline bool operator!=(const ScalarType &lhs, const ScalarType &rhs) noexcept {
    return std::addressof(lhs) != std::addressof(rhs);
}

// Implementation of type getting mechanics

namespace dtl {

#define ROUGHPY_MAKE_TYPE_ID_OF(TYPE, NAME)           \
    template <>                                       \
    struct type_id_of_impl<TYPE> {                    \
        static const std::string &get_id() noexcept { \
            static const std::string type_id(NAME);   \
            return type_id;                           \
        }                                             \
    }

ROUGHPY_MAKE_TYPE_ID_OF(char, "i8");
ROUGHPY_MAKE_TYPE_ID_OF(unsigned char, "u8");
ROUGHPY_MAKE_TYPE_ID_OF(short, "i16");
ROUGHPY_MAKE_TYPE_ID_OF(unsigned short, "u16");
ROUGHPY_MAKE_TYPE_ID_OF(int, "i32");
ROUGHPY_MAKE_TYPE_ID_OF(unsigned int, "u32");
ROUGHPY_MAKE_TYPE_ID_OF(long long, "i64");
ROUGHPY_MAKE_TYPE_ID_OF(unsigned long long, "u64");
ROUGHPY_MAKE_TYPE_ID_OF(signed_size_type_marker, "isize");
ROUGHPY_MAKE_TYPE_ID_OF(unsigned_size_type_marker, "usize");

ROUGHPY_MAKE_TYPE_ID_OF(float, "f32");
ROUGHPY_MAKE_TYPE_ID_OF(double, "f64");

// Long is silly. On Win64 it is 32 bits (because, Microsoft) on Unix, it is 64 bits
template <>
struct type_id_of_impl<long>
    : public std::conditional_t<(sizeof(long) == sizeof(int)),
                                type_id_of_impl<int>,
                                type_id_of_impl<long long>> {};

#undef ROUGHPY_MAKE_TYPE_ID_OF

template <typename ScalarImpl>
struct scalar_type_holder {
    static const ScalarType* get_type() noexcept;
};

template <typename ScalarImpl>
struct scalar_type_holder<ScalarImpl &> : scalar_type_holder<ScalarImpl> {};

template <typename ScalarImpl>
struct scalar_type_holder<const ScalarImpl &> : scalar_type_holder<ScalarImpl> {};

template <>
ROUGHPY_SCALARS_EXPORT const ScalarType *scalar_type_holder<float>::get_type() noexcept;

template <>
ROUGHPY_SCALARS_EXPORT const ScalarType *scalar_type_holder<double>::get_type() noexcept;

//template <>
//ROUGHPY_SCALARS_EXPORT const ScalarType *scalar_type_holder<rational_scalar_type>::get_type() noexcept;

}// namespace dtl

template <typename T>
const ScalarType *ScalarType::of() {
    return dtl::scalar_type_holder<T>::get_type();
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_TYPE_H_
