//
// Created by user on 28/02/23.
//

#include "scalar_stream.h"

#include "scalar_pointer.h"
#include "scalar_type.h"
#include "scalar_array.h"
#include "scalar.h"

using namespace rpy::scalars;

ScalarStream::ScalarStream() : m_stream(), m_elts_per_row(0), p_type(nullptr) {
}

ScalarStream::ScalarStream(const ScalarType *type)
    : m_stream(), m_elts_per_row(0), p_type(type) {
}
ScalarStream::ScalarStream(ScalarPointer base, std::vector<dimn_t> shape) {
    if (!base.is_null()) {
        p_type = base.type();
        if (p_type == nullptr) {
            throw std::runtime_error("missing type");
        }
        if (shape.empty()) {
            throw std::runtime_error("strides cannot be empty");
        }

        const auto *ptr = static_cast<const char *>(base.ptr());
        const auto itemsize = p_type->itemsize();

        dimn_t rows = shape[0];
        dimn_t cols = (shape.size() > 1) ? shape[1] : 1;

        m_elts_per_row.push_back(cols);

        dimn_t stride = cols * itemsize;
        m_stream.reserve(rows);

        for (dimn_t i = 0; i < rows; ++i) {
            m_stream.push_back(ptr);
            ptr += stride;
        }
    }
}

dimn_t ScalarStream::col_count(dimn_t i) const noexcept {
    if (m_elts_per_row.size() == 1) {
        return m_elts_per_row[0];
    }

    assert(m_elts_per_row.size() > 1);
    assert(i < m_elts_per_row.size());
    return m_elts_per_row[i];
}

ScalarArray ScalarStream::operator[](dimn_t row) const noexcept {
    return {ScalarPointer(p_type, m_stream[row]), col_count(row)};
}
Scalar ScalarStream::operator[](std::pair<dimn_t, dimn_t> index) const noexcept {
    auto first = operator[](index.first);
    return first[index.second];
}

void ScalarStream::set_elts_per_row(dimn_t num_elts) noexcept {
    if (m_elts_per_row.size() > 1) {
        m_elts_per_row.clear();
        m_elts_per_row.push_back(num_elts);
    } else if (m_elts_per_row.size() == 1) {
        m_elts_per_row[0] = num_elts;
    } else {
        m_elts_per_row.push_back(num_elts);
    }
}
void ScalarStream::reserve_size(dimn_t num_rows) {
    m_stream.reserve(num_rows);
}
void ScalarStream::push_back(const ScalarPointer &data) {
    assert(m_elts_per_row.size() == 1 && m_elts_per_row[0] > 0);
    m_stream.push_back(data.ptr());
}
void ScalarStream::push_back(const ScalarArray &data) {
    if (m_elts_per_row.size() == 1) {
        m_stream.push_back(data.ptr());
        if (data.size() != m_elts_per_row[0]) {
            m_elts_per_row.reserve(m_stream.size() + 1);
            m_elts_per_row.resize(m_stream.size(), m_elts_per_row[0]);
            m_elts_per_row.push_back(data.size());
        }
    } else {
        m_stream.push_back(data.ptr());
        m_elts_per_row.push_back(data.size());
    }
}

void ScalarStream::set_ctype(const scalars::ScalarType *type) noexcept {
    m_stream.clear();
    p_type = type;
}
