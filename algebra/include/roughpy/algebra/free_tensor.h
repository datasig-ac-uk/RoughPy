//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_H
#define ROUGHPY_ALGEBRA_FREE_TENSOR_H

#include "algebra.h"


namespace rpy { namespace algebra {


class FreeTensorMultiplication : public Multiplication {

public:

    virtual devices::Kernel get_kernel(string_view suffix) const;

};



class FreeTensor : public Algebra {




};


}}

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_H
