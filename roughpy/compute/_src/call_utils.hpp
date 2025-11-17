#ifndef ROUGHPY_COMPUTE_INCLUDE_ROUGHPY_COMPUTE_CAL_UTILS_HPP
#define ROUGHPY_COMPUTE_INCLUDE_ROUGHPY_COMPUTE_CAL_UTILS_HPP

#include "py_headers.h"

#include <iterator>
#include <exception>
#include <type_traits>

#include <roughpy_compute/common/basis.hpp>
#include <roughpy_compute/common/scalars.hpp>
#include <roughpy_compute/dense/views.hpp>


#include "call_config.hpp"
#include "py_obj_handle.hpp"

#include "object_arrays/object_compute_context.hpp"
#include "object_arrays/object_array_iterator.hpp"


namespace rpy::compute {

struct DegreeBounds {
    int32_t max_degree = -1;
    int32_t min_degree = 0;
};


struct CallConfig
{
    DegreeBounds* degree_bounds = nullptr;
    BasisBase const* const* basis_data = nullptr;
    void* ops;
};



namespace call_utils_dtl {

template <typename Iter>
inline constexpr bool is_object_iter_v = std::is_same_v<
        std::remove_cv_t<typename std::iterator_traits<Iter>::value_type>,
        PyObject*>;

template <typename Iter>
inline constexpr bool is_const_iter_v = std::is_const_v<
    std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>;

} // namespace call_utils_dtl

template <npy_intp CoreDims_, npy_intp... ArgBasisMapping>
struct ComputeCallFunctor {
    static constexpr npy_intp CoreDims = CoreDims_;
    static constexpr npy_intp n_args = sizeof...(ArgBasisMapping);

    static constexpr npy_intp arg_basis_mapping[n_args] = { ArgBasisMapping... };

    CallConfig const* config_;

    explicit constexpr ComputeCallFunctor(CallConfig const& config) noexcept
        : config_(&config)
    {}

    constexpr ComputeCallFunctor(ComputeCallFunctor&& other) noexcept
        : config_(other.config_)
    {}

    template <typename Iter>
    constexpr auto make_vector_fragment(npy_intp arg_id, Iter iter) const noexcept
    {
        auto* basis = this->config_->basis_data[arg_basis_mapping[arg_id]];

        using Adapter = std::conditional_t<
                call_utils_dtl::is_object_iter_v<Iter>,
                std::conditional_t<
                        call_utils_dtl::is_const_iter_v<Iter>,
                        ConstObjectArrayIterator<Iter>,
                        MutableObjectArrayIterator<Iter>>,
                Iter>;
        using View = DenseVectorFragment<Adapter>;

        return View(
            Adapter(std::move(iter)),
            basis->size()
            );
    }

    template <typename Iter>
    constexpr auto make_tensor_view(npy_intp arg_id, Iter iter) const noexcept
    {
        using Adapter = std::conditional_t<
                call_utils_dtl::is_object_iter_v<Iter>,
                std::conditional_t<
                        call_utils_dtl::is_const_iter_v<Iter>,
                        ConstObjectArrayIterator<Iter>,
                        MutableObjectArrayIterator<Iter>>,
                Iter>;
        using View = DenseTensorView<Adapter>;

        auto* basis = static_cast<TensorBasis const*>(
                this->config_->basis_data[arg_basis_mapping[arg_id]]
        );
        const auto& bounds = this->config_->degree_bounds[arg_id];

        return View(
            Adapter(std::move(iter)),
            *basis,
            bounds.min_degree,
            bounds.max_degree
        );
    }

    template <typename Iter>
    constexpr auto make_lie_view(npy_intp arg_id, Iter iter) const noexcept
    {
        using Adapter = std::conditional_t<
                call_utils_dtl::is_object_iter_v<Iter>,
                std::conditional_t<
                        call_utils_dtl::is_const_iter_v<Iter>,
                        ConstObjectArrayIterator<Iter>,
                        MutableObjectArrayIterator<Iter>>,
                Iter>;
        using View = DenseTensorView<Adapter>;

        auto* basis = static_cast<LieBasis const*>(
                this->config_->basis_data[arg_basis_mapping[arg_id]]
        );
        const auto& bounds = this->config_->degree_bounds[arg_id];

        return View(
            Adapter(std::move(iter)),
            *basis,
            bounds.min_degree,
            bounds.max_degree
        );
    }


    template <typename S>
    constexpr auto get_context(const S& arg) const noexcept
    {
        if constexpr (!std::is_same_v<std::remove_cv_t<S>, ObjectRef>) {
            // is not a python object
            return scalars::Traits<S> {};
        } else {
            // is a python object, needs a python context
            return ObjectComputeContext(Py_TYPE(arg.obj()));
        }
    }

};







struct LieBasisArrayHolder
{
    PyArrayObject* degree_begin = nullptr;
    PyArrayObject* data = nullptr;

    LieBasisArrayHolder() = default;

    LieBasisArrayHolder(LieBasisArrayHolder&& old) noexcept
        : degree_begin(old.degree_begin), data(old.data)
    {
        old.degree_begin = nullptr;
        old.data = nullptr;
    }

    ~LieBasisArrayHolder()
    {
        Py_XDECREF(degree_begin);
        Py_XDECREF(data);
    }

    explicit operator bool() const noexcept
    {
        return degree_begin != nullptr && data != nullptr;
    }

};

bool update_algebra_params(CallConfig& config, npy_intp n_args, npy_intp const* arg_basis_mapping);

PyObjHandle to_basis(PyObject* basis_obj, TensorBasis& basis);
LieBasisArrayHolder to_basis(PyObject* basis_obj, LieBasis& basis);

} // namespace rpy::compute

#endif// ROUGHPY_COMPUTE_INCLUDE_ROUGHPY_COMPUTE_CAL_UTILS_HPP
