//
// Created by user on 28/02/23.
//

#include "scalar_array.h"

rpy::scalars::ScalarArray::ScalarArray(rpy::scalars::ScalarArray &&other) noexcept
    : ScalarPointer(other), m_size(other.m_size) {
    /*
     * It doesn't really matter for this class, but various
     * derived classes will need to make sure that ownership
     * is transferred on move. We reset the p_data and size
     * to null values so that, in derived classes, the
     * destructor does not free the data while it is still
     * in use.
     */
    other.p_data = nullptr;
    other.m_size = 0;
}
rpy::scalars::ScalarArray &rpy::scalars::ScalarArray::operator=(rpy::scalars::ScalarArray &&other) noexcept {
    if (std::addressof(other) != this) {
        p_type = other.p_type;
        p_data = other.p_data;
        m_size = other.m_size;

        /*
         * It doesn't really matter for this class, but various
         * derived classes will need to make sure that ownership
         * is transferred on move. We reset the p_data and size
         * to null values so that, in derived classes, the
         * destructor does not free the data while it is still
         * in use.
         */
        other.p_data = nullptr;
        other.m_size = 0;
    }
    return *this;
}
