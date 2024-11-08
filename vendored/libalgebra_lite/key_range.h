//
// Created by user on 08/08/22.
//

#ifndef LIBALGEBRA_LITE_KEY_RANGE_H
#define LIBALGEBRA_LITE_KEY_RANGE_H

#include "implementation_types.h"
#include "basis_traits.h"

namespace lal {

template <typename Basis>
class key_range
{
    using traits = basis_trait<Basis>;
    using key_type = typename traits::key_type;

    const Basis* p_basis;
    const key_type start_key;
    const key_type end_key;


public:

    key_range(const Basis* p_basis, key_type begin, key_type end);
    key_range(const Basis* p_basis, std::pair<key_type, key_type> keys);

    dimn_t begin_idx() const noexcept
    { return traits::key_to_index(*p_basis, start_key); }
    dimn_t end_idx() const noexcept
    { return traits::key_to_index(*p_basis, end_key); }

};

} // namespace lal


#endif //LIBALGEBRA_LITE_KEY_RANGE_H
