#ifndef ROUGHPY_ALGEBRA_CONTEXT_H_
#define ROUGHPY_ALGEBRA_CONTEXT_H_

#include "algebra_fwd.h"
#include "context_fwd.h"

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

#include <roughpy/core/helpers.h>
#include <roughpy/core/macros.h>
#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/scalar_stream.h>
#include <roughpy/scalars/scalar_type.h>

#include <memory>
#include <vector>
#include <string>

#include "lie_basis.h"
#include "tensor_basis.h"
#include "free_tensor.h"
#include "lie.h"
#include "shuffle_tensor.h"

namespace rpy {
namespace algebra {

struct SignatureData {
    scalars::ScalarStream data_stream;
    std::vector<const key_type *> key_stream;
    VectorType vector_type;
};

struct DerivativeComputeInfo {
    Lie logsig_of_interval;
    Lie perturbation;
};

struct VectorConstructionData {
    scalars::KeyScalarArray data;
    VectorType vector_type = VectorType::Sparse;
};

class ROUGHPY_ALGEBRA_EXPORT ContextBase
    : public boost::intrusive_ref_counter<ContextBase> {
    MaybeOwned<const dimn_t> p_lie_sizes;
    MaybeOwned<const dimn_t> p_tensor_sizes;

    deg_t m_width;
    deg_t m_depth;

protected:
    ContextBase(deg_t width,
                deg_t depth,
                const dimn_t *lie_sizes,
                const dimn_t *tensor_sizes);

public:
    virtual ~ContextBase();

    deg_t width() const noexcept { return m_width; }
    deg_t depth() const noexcept { return m_depth; }

    dimn_t lie_size(deg_t deg) const noexcept;
    dimn_t tensor_size(deg_t deg) const noexcept;
};

class ROUGHPY_ALGEBRA_EXPORT Context : public ContextBase {
    const scalars::ScalarType *p_ctype;
    std::string m_ctx_backend;

protected:
    explicit Context(
        deg_t width, deg_t depth,
        const scalars::ScalarType *ctype,
        std::string&& context_backend,
        const dimn_t *lie_sizes = nullptr,
        const dimn_t *tensor_sizes = nullptr)
        : ContextBase(width, depth, lie_sizes, tensor_sizes),
          p_ctype(ctype),
          m_ctx_backend(std::move(context_backend))
    {
    }


public:

    const scalars::ScalarType *ctype() const noexcept { return p_ctype; }
    const std::string &backend() const noexcept { return m_ctx_backend; }

    virtual context_pointer get_alike(deg_t new_depth) const = 0;
    virtual context_pointer get_alike(const scalars::ScalarType *new_ctype) const = 0;
    virtual context_pointer get_alike(deg_t new_depth, const scalars::ScalarType *new_ctype) const = 0;
    virtual context_pointer get_alike(deg_t new_width, deg_t new_depth, const scalars::ScalarType *new_ctype) const = 0;

    virtual bool check_compatible(const Context &other_ctx) const noexcept;

    virtual Basis get_lie_basis() const = 0;
    virtual Basis get_tensor_basis() const = 0;

    virtual FreeTensor convert(const FreeTensor &arg, optional<VectorType> new_vec_type) const = 0;
    virtual ShuffleTensor convert(const ShuffleTensor &arg, optional<VectorType> new_vec_type) const = 0;
    virtual Lie convert(const Lie &arg, optional<VectorType> new_vec_type) const = 0;

    virtual FreeTensor construct_free_tensor(const VectorConstructionData &arg) const = 0;
    virtual ShuffleTensor construct_shuffle_tensor(const VectorConstructionData &arg) const = 0;
    virtual Lie construct_lie(const VectorConstructionData &arg) const = 0;

    virtual UnspecifiedAlgebraType construct(AlgebraType type, const VectorConstructionData& data) const = 0;


    FreeTensor zero_free_tensor(VectorType vtype) const;
    ShuffleTensor zero_shuffle_tensor(VectorType vtype) const;
    Lie zero_lie(VectorType vtype) const;

protected:
    void lie_to_tensor_fallback(FreeTensor &result, const Lie &arg) const;
    void tensor_to_lie_fallback(Lie &result, const FreeTensor &arg) const;

public:
    virtual FreeTensor lie_to_tensor(const Lie &arg) const = 0;
    virtual Lie tensor_to_lie(const FreeTensor &arg) const = 0;

protected:
    void cbh_fallback(FreeTensor &collector, const std::vector<Lie> &lies) const;

public:
    virtual Lie cbh(const std::vector<Lie> &lies, VectorType vtype) const;
    virtual Lie cbh(const Lie& left, const Lie& right, VectorType vtype) const;

    virtual FreeTensor to_signature(const Lie &log_signature) const;
    virtual FreeTensor signature(const SignatureData &data) const = 0;
    virtual Lie log_signature(const SignatureData &data) const = 0;

    virtual FreeTensor sig_derivative(const std::vector<DerivativeComputeInfo> &info,
                                      VectorType vtype) const = 0;
};

ROUGHPY_ALGEBRA_EXPORT
base_context_pointer get_base_context(deg_t width, deg_t depth);

ROUGHPY_ALGEBRA_EXPORT
context_pointer get_context(deg_t width, deg_t depth, const scalars::ScalarType *ctype,
                            const std::vector<std::pair<std::string, std::string>> &preferences = {});



class ROUGHPY_ALGEBRA_EXPORT ContextMaker {
public:
    using preference_list = std::vector<std::pair<std::string, std::string>>;

    virtual ~ContextMaker() = default;
    virtual bool can_get(deg_t width, deg_t depth, const scalars::ScalarType *ctype,
                         const preference_list &preferences) const;
    virtual context_pointer get_context(deg_t width, deg_t depth, const scalars::ScalarType *ctype,
                                        const preference_list &preferences) const = 0;
    virtual optional<base_context_pointer> get_base_context(deg_t width, deg_t depth) const = 0;
};

ROUGHPY_ALGEBRA_EXPORT
const ContextMaker *register_context_maker(std::unique_ptr<ContextMaker> maker);

template <typename Maker>
class RegisterMakerHelper {
    const ContextMaker *maker = nullptr;

public:
    template <typename... Args>
    explicit RegisterMakerHelper(Args &&...args) {
        maker = register_context_maker(
            std::unique_ptr<ContextMaker>(
                new Maker(std::forward<Args>(args)...)));
    }
};

#define RPY_ALGEBRA_DECLARE_CTX_MAKER(MAKER, ...) \
    static RPY_USED RegisterMakerHelper<MAKER> rpy_static_algebra_maker_decl_##MAKER = RegisterMakerHelper<MAKER>(__VA_ARGS__)

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_CONTEXT_H_
