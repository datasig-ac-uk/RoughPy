//
// Created by sam on 11/23/23.
//
#include "devices/core.h"


std::ostream& rpy::devices::operator<<(std::ostream& os, const TypeInfo& info)
{
    switch (info.code) {
        case TypeCode::Int: os << "int" << info.bytes * CHAR_BIT;
            break;
        case TypeCode::UInt: os << "uint" << info.bytes * CHAR_BIT;
            break;
        case TypeCode::Float: os << "float" << info.bytes * CHAR_BIT;
            break;
        case TypeCode::OpaqueHandle: os << "opaque";
            break;
        case TypeCode::BFloat: os << "bfloat" << info.bytes * CHAR_BIT;
            break;
        case TypeCode::Complex: os << "complex" << info.bytes * CHAR_BIT;
            break;
        case TypeCode::Bool: os << "bool";
            break;
        case TypeCode::Rational: os << "Rational";
            break;
        case TypeCode::ArbitraryPrecisionInt: os << "mp_int";
            break;
        case TypeCode::ArbitraryPrecisionUInt: os << "mp_uint";
            break;
        case TypeCode::ArbitraryPrecisionFloat: os << "mp_float";
            break;
        case TypeCode::ArbitraryPrecisionComplex: os << "mp_complex";
            break;
        case TypeCode::ArbitraryPrecisionRational: os << "Rational";
            break;
        case TypeCode::APRationalPolynomial: os << "PolyRational";
            break;
    }

    return os;
}
