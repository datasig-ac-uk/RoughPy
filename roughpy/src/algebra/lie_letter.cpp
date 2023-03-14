#include "lie_letter.h"

#include <ostream>

using namespace rpy;

std::ostream &python::operator<<(std::ostream &os, const python::PyLieLetter &let) {
    return os << let.m_data;
}
