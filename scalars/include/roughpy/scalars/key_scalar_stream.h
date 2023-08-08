//
// Created by sam on 08/08/23.
//

#ifndef ROUGHPY_SCALARS_KEY_SCALAR_STREAM_H
#define ROUGHPY_SCALARS_KEY_SCALAR_STREAM_H

#include "scalars_fwd.h"
#include "scalar_array.h"
#include "key_scalar_array.h"
#include "scalar_stream.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

namespace rpy { namespace scalars {


class RPY_EXPORT KeyScalarStream : public ScalarStream {
    std::vector<const key_type*> m_key_stream;


public:

    KeyScalarStream();
    KeyScalarStream(const KeyScalarStream&);
    KeyScalarStream(KeyScalarStream&&) noexcept;

    KeyScalarStream& operator=(const KeyScalarStream&);
    KeyScalarStream& operator=(KeyScalarStream&&) noexcept;

    RPY_NO_DISCARD
    KeyScalarArray operator[](dimn_t row) const noexcept;

    void reserve_size(dimn_t num_rows);

    void push_back(const ScalarPointer& scalar_ptr, const key_type* key_ptr=nullptr);
    void push_back(const ScalarArray& scalar_data, const key_type* key_ptr=nullptr);

};



}}


#endif// ROUGHPY_SCALARS_KEY_SCALAR_STREAM_H
