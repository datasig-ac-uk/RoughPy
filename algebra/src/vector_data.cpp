//
// Created by sam on 22/03/24.
//

#include "vector.h"

#include "kernels/kernel.h"
#include "key_algorithms.h"
#include <roughpy/scalars/algorithms.h>

using namespace rpy;
using namespace rpy::algebra;

void VectorData::reserve(dimn_t dim)
{
    if (dim <= capacity()) { return; }

    auto new_buffer = m_scalar_buffer.type()->allocate(dim);
    scalars::algorithms::copy(new_buffer, m_scalar_buffer);

    if (!m_key_buffer.empty()) {
        KeyArray new_key_buffer(m_key_buffer.device(), dim);
        algorithms::copy(new_key_buffer, m_key_buffer);
        std::swap(m_key_buffer, new_key_buffer);
    }

    std::swap(m_scalar_buffer, new_buffer);
}

void VectorData::resize(dimn_t dim)
{
    reserve(dim);
    m_size = dim;
}

void VectorData::insert_element(
        dimn_t index,
        dimn_t next_size,
        const BasisKey& key,
        scalars::Scalar value
)
{}
void VectorData::delete_element(dimn_t index)
{
    RPY_DBG_ASSERT(index < m_scalar_buffer.size());
    auto scalar_view = m_scalar_buffer.mut_view();
    m_size -= 1;
    for (dimn_t i = index; i < m_size; ++i) {
        scalar_view[i] = scalar_view[i + 1];
    }

    if (!m_key_buffer.empty()) {
        auto key_view = m_key_buffer.as_mut_slice();
        for (dimn_t i = index; i < m_size; ++i) {
            key_view[i] = std::move(key_view[i + 1]);
        }
    }
}

std::unique_ptr<VectorData> VectorData::make_dense(const Basis* basis) const
{
    RPY_CHECK(basis->is_ordered() && sparse());

    const auto scalar_device = m_scalar_buffer.device();
    const auto key_device = m_key_buffer.device();

    auto dense_data = std::make_unique<VectorData>(m_scalar_buffer.type());

    dimn_t dimension;
    if (key_device->is_host()) {
        /*
         * If we're on host, then the keys might not be an array of indices, so
         * we need to do the transformation into indices first, then look for
         * the maximum value.
         */
        dimension = 0;
        dimn_t index;
        for (auto& key : dense_data->m_key_buffer.as_mut_slice()) {
            index = basis->to_index(key);
            if (index > dimension) { dimension = index; }
            key = BasisKey(index);
        }

    } else {
        /*
         * On device, all of the key should be indices so we can just use the
         * algorithms max to find the maximum index that exists in the vector.
         */
        dimension = algorithms::max(m_key_buffer);
    }

    auto resize_dim = basis->dense_dimension(dimension);

    dense_data->resize(resize_dim);

    /*
     * We have already made sure that the buffer only contains indices so we can
     * call a kernel to write the data into the dense array. The only remaining
     * tricky part is to make sure the keys live on the same device as the
     * scalars. We can fix this by copying the data to the device if necessary.
     */

    devices::Buffer key_buffer;
    if (scalar_device == key_device) {
        key_buffer = dense_data->m_key_buffer.mut_buffer();
        ;
    } else {
        dense_data->m_key_buffer.mut_buffer().to_device(
                key_buffer,
                scalar_device
        );
    }

    auto kernel = dtl::get_kernel(
            "sparse_write",
            scalar_type()->id(),
            "ds",
            scalar_device
    );
    devices::KernelLaunchParams params(
            devices::Size3{m_size},
            devices::Dim3{1}
    );
    RPY_CHECK(kernel);

    (*kernel)(
            params,
            devices::Buffer(dense_data->mut_scalar_buffer()),
            devices::Buffer(key_buffer),
            devices::Buffer(m_scalar_buffer.buffer())
    );

    return dense_data;
}

std::unique_ptr<VectorData> VectorData::make_sparse(const Basis* basis) const
{
    RPY_CHECK(!sparse());

    auto sparse_data = std::make_unique<VectorData>(*this);

    KeyArray keys(m_size);
    {
        auto key_slice = keys.as_mut_slice();
        for (auto [i, k] : views::enumerate(key_slice)) { k = i; }
    }

    sparse_data->m_key_buffer = std::move(keys);

    return sparse_data;
}
