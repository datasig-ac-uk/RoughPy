//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALARS_SCALAR_VECTOR_H
#define ROUGHPY_SCALARS_SCALAR_VECTOR_H


#include "scalars_fwd.h"
#include <roughpy/core/smart_ptr.h>
#include <roughpy/platform/alloc.h>

#include <roughpy/devices/core.h>
#include <roughpy/devices/buffer.h>
#include <roughpy/devices/type.h>
#include <roughpy/devices/device_handle.h>
#include <roughpy/devices/host_device.h>


#include "scalar_array.h"

namespace rpy { namespace scalars {


namespace dtl {

class ROUGHPY_SCALARS_SCALARS_EXPORT VectorData : public platform::SmallObjectBase,
                                          public RcBase<VectorData>
{
    ScalarArray m_scalar_buffer{};
    dimn_t m_size = 0;

public:
    void set_size(dimn_t size)
    {
        RPY_CHECK(size <= m_scalar_buffer.size());
        m_size = size;
    }

    VectorData() = default;

    explicit VectorData(TypePtr type, dimn_t size)
        : m_scalar_buffer(type, size),
          m_size(size)
    {}

    explicit VectorData(ScalarArray&& scalars)
        : m_scalar_buffer(std::move(scalars)),
          m_size(scalars.size())
    {}

    explicit VectorData(TypePtr type) : m_scalar_buffer(type) {}

    void reserve(dimn_t dim);
    void resize(dimn_t dim);

    RPY_NO_DISCARD dimn_t capacity() const noexcept
    {
        return m_scalar_buffer.size();
    }
    RPY_NO_DISCARD dimn_t size() const noexcept { return m_size; }

    RPY_NO_DISCARD bool empty() const noexcept
    {
        return m_scalar_buffer.empty();
    }

    RPY_NO_DISCARD TypePtr scalar_type() const noexcept
    {
        return m_scalar_buffer.type();
    }

    RPY_NO_DISCARD devices::Buffer& mut_scalar_buffer() noexcept
    {
        return m_scalar_buffer.mut_buffer();
    }
    RPY_NO_DISCARD const devices::Buffer& scalar_buffer() const noexcept
    {
        return m_scalar_buffer.buffer();
    }

    RPY_NO_DISCARD ScalarArray& mut_scalars() noexcept
    {
        return m_scalar_buffer;
    }
    RPY_NO_DISCARD const ScalarArray& scalars() const noexcept
    {
        return m_scalar_buffer;
    }

    void insert_element(
            dimn_t index,
            dimn_t next_size,
            Scalar value
    );
    void delete_element(dimn_t index);

};

class ScalarVectorIterator
{

};
}


class ROUGHPY_SCALARS_SCALARS_EXPORT ScalarVector
{
    using VectorDataPtr = Rc<dtl::VectorData>;

    VectorDataPtr p_base = nullptr;
    VectorDataPtr p_fibre = nullptr;

    friend class MutableVectorElement;

public:
    using iterator = dtl::ScalarVectorIterator;
    using const_iterator = dtl::ScalarVectorIterator;
    using value_type = Scalar;
    using reference = ScalarRef;
    using const_reference = ScalarCRef;

protected:
    void resize_dim(dimn_t new_dim);

    RPY_NO_DISCARD dimn_t buffer_size() const noexcept
    {
        return fast_is_zero() ? 0 : p_base->size();
    }

    void set_zero() const noexcept;

    ScalarVector(VectorDataPtr base, VectorDataPtr fibre)
        : p_base(std::move(base)), p_fibre(std::move(fibre))
    {}

    ScalarArray& mut_scalars() const noexcept
    {
        RPY_DBG_ASSERT(p_base != nullptr);
        return p_base->mut_scalars();
    }
    const ScalarArray& scalars() const noexcept
    {
        RPY_DBG_ASSERT(p_base != nullptr);
        return p_base->scalars();
    }

public:

    

    ScalarVector(TypePtr scalar_type, dimn_t size=0)
        : p_base(new dtl::VectorData(std::move(scalar_type), size)),
          p_fibre(nullptr)
    {}

    RPY_NO_DISCARD bool fast_is_zero() const noexcept
    {
        return p_base == nullptr || p_base->empty();
    }

    RPY_NO_DISCARD ScalarVector base() const noexcept;
    RPY_NO_DISCARD ScalarVector fibre() const noexcept;

    RPY_NO_DISCARD
    devices::Device device() const noexcept
    {
        return fast_is_zero() ? static_cast<devices::Device>(devices::get_host_device())
                              : p_base->scalars().device();
    }

    RPY_NO_DISCARD
    TypePtr scalar_type() const noexcept
    {
        RPY_DBG_ASSERT(p_base != nullptr);
        return p_base->scalar_type();
    }


    RPY_NO_DISCARD dimn_t dimension() const noexcept;
    RPY_NO_DISCARD dimn_t size() const noexcept;
    RPY_NO_DISCARD bool is_zero() const noexcept;

    RPY_NO_DISCARD const_reference get(dimn_t index) const;
    RPY_NO_DISCARD reference get_mut(dimn_t index);

    RPY_NO_DISCARD const_iterator begin() const noexcept;
    RPY_NO_DISCARD const_iterator end() const noexcept;

    template <typename V>
    RPY_NO_DISCARD
    enable_if_t<is_base_of_v<ScalarVector, V>, V> borrow() const
    {
        return V(p_base, p_fibre);
    }

    template <typename V>
    RPY_NO_DISCARD enable_if_t<is_base_of_v<ScalarVector, V>,V>
    borrow_mut()
    {
        return V(p_base, p_fibre);
    }


    RPY_NO_DISCARD ScalarVector uminus() const;

    RPY_NO_DISCARD ScalarVector add(const ScalarVector& other) const;

    RPY_NO_DISCARD ScalarVector sub(const ScalarVector& other) const;

    RPY_NO_DISCARD ScalarVector left_smul(const Scalar& scalar) const;
    RPY_NO_DISCARD ScalarVector right_smul(const Scalar& scalar) const;
    RPY_NO_DISCARD ScalarVector sdiv(const Scalar& scalar) const;


    ScalarVector& add_inplace(const ScalarVector& other);
    ScalarVector& sub_inplace(const ScalarVector& other);
    ScalarVector& left_smul_inplace(const Scalar& other);
    ScalarVector& right_smul_inplace(const Scalar& other);
    ScalarVector& sdiv_inplace(const Scalar& other);

    ScalarVector& add_scal_mul(const ScalarVector& other, const Scalar& scalar);
    ScalarVector& sub_scal_mul(const ScalarVector& other, const Scalar& scalar);

    ScalarVector& add_scal_div(const ScalarVector& other, const Scalar& scalar);
    ScalarVector& sub_scal_div(const ScalarVector& other, const Scalar& scalar);

    RPY_NO_DISCARD bool operator==(const ScalarVector& other) const;
    RPY_NO_DISCARD bool operator!=(const ScalarVector& other) const
    {
        return !operator==(other);
    }

};

}}


#endif //ROUGHPY_SCALARS_SCALAR_VECTOR_H
