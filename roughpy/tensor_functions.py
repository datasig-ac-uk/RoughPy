"""
This module provides additional tensor functions that are not yet implemented in
the RoughPy core library.

The implementation of the functions in this module are far from optimal, but
should serve as both a temporary implementation and a demonstration of how one
can build on top of the RoughPy core library.
"""


import functools
from collections import defaultdict
from typing import Any, Union, TypeVar

import roughpy as rp


def tensor_word_factory(basis):
    """
    Create a factory function that constructs tensor words objects.

    Since the tensor words are specific to their basis, the basis is needed to
    construct the words. This function creates a factory from the correct basis
    to make tensor words that correspond. The arguments to the factory function
    are just a sequence of letters in the same order as they will appear in the
    tensor word.

    :param basis: RoughPy tensor basis object.
    :return: function from a sequence of letters to tensor words
    """
    width = basis.width
    depth = basis.depth

    # noinspection PyArgumentList
    def factory(*args):
        return rp.TensorKey(*args, width=width, depth=depth)

    return factory


class TensorTensorProduct:
    """
    External tensor product of two free tensors (or shuffle tensors).

    This is an intermediate container that is used to implement some of the
    tensor functions such as Log.
    """

    data: 'dict[tuple[rp.TensorKey], Any]'
    ctx: 'rp.Context'

    def __init__(self, data, ctx=None):

        if isinstance(data, (tuple, list)):
            assert len(data) == 2

            assert isinstance(data[0], (rp.FreeTensor, rp.ShuffleTensor))
            assert isinstance(data[1], (rp.FreeTensor, rp.ShuffleTensor))

            if ctx is not None:
                self.ctx = ctx
                assert data[0].context == ctx
                assert data[1].context == ctx
            else:
                self.ctx = data[0].context
                assert self.ctx == data[2].context

            self.data = odata = {}
            for lhs in data[0]:
                for rhs in data[1]:
                    odata[(lhs.key(), rhs.key())] = lhs.value() * rhs.value()

            # self.data = {k: v for k, v in odata.items() if v != 0}

        elif isinstance(data, (dict, defaultdict)):
            self.data = {k: v for k, v in data.items() if v != 0}
            assert ctx is not None
            self.ctx = ctx

    def __str__(self):
        return " ".join([
            '{',
            *(f"{v}{k}" for k, v in sorted(self.data.items(),
                                           key=lambda t: tuple(
                                               map(lambda r: tuple(
                                                   r.to_letters()), t[0])))),
            '}'
        ])

    def __repr__(self):
        return " ".join([
            '{',
            *(f"{v}{k}" for k, v in sorted(self.data.items(),
                                           key=lambda t: tuple(
                                               map(lambda r: tuple(
                                                   r.to_letters()), t[0])))),
            '}'
        ])

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float, rp.Scalar)):
            if scalar == 0:
                data = {}
            else:
                data = {k: v * scalar for k, v in self.data.items()}
            return TensorTensorProduct(data, self.ctx)

        return NotImplemented

    def __add__(self, other):
        if isinstance(other, TensorTensorProduct):
            assert self.ctx == other.ctx
            new_data = defaultdict(lambda: 0)

            # Do this to force a deep copy of the values from self
            for k, v in self.data.items():
                new_data[k] += v

            for k, v in other.data.items():
                new_data[k] += v

            return TensorTensorProduct(
                {k: v for k, v in new_data.items() if v != 0}, self.ctx)

        return NotImplemented

    def add_scal_prod(self, other, scalar):
        my_data = self.data
        for k, v in other.data.items():

            val = v * scalar
            if k in self.data:
                my_data[k] += val
            else:
                my_data[k] = val

        # print("asp", other, scalar, self)
        self.data = {k: v for k, v in self.data.items() if v != 0}
        return self

    def add_scal_div(self, other, scalar):
        my_data = self.data
        for k, v in other.data.items():
            if k in self.data:
                my_data[k] += v / scalar
            else:
                my_data[k] = v / scalar

        return self

    def sub_scal_div(self, other, scalar):
        my_data = self.data
        for k, v in other.data.items():
            if k in self.data:
                my_data[k] -= v / scalar
            else:
                my_data[k] = -v / scalar

        return self


def _concat_product(a, b):
    out = defaultdict(lambda: 0)

    for k1, v1 in a.data.items():
        for k2, v2 in b.data.items():
            out[tuple(i * j for i, j in zip(k1, k2))] += v1 * v2

    return TensorTensorProduct({k: v for k, v in out.items() if v != 0}, a.ctx)


# noinspection PyUnresolvedReferences
def _adjoint_of_word(word: rp.TensorKey, ctx: rp.Context) \
        -> TensorTensorProduct:
    word_factory = tensor_word_factory(word.basis())
    letters = word.to_letters()
    if not letters:
        return TensorTensorProduct({(word_factory(),) * 2: 1}, ctx)

    letters_adj = [
        TensorTensorProduct({(word_factory(letter), word_factory()): 1,
                             (word_factory(), word_factory(letter)): 1}, ctx)
        for letter in word.to_letters()]
    return functools.reduce(_concat_product, letters_adj)


def _adjoint_of_shuffle(
        tensor: Union[rp.FreeTensor, rp.ShuffleTensor]) -> TensorTensorProduct:
    # noinspection PyUnresolvedReferences
    ctx = tensor.context
    out = TensorTensorProduct(defaultdict(lambda: 0), ctx)

    for item in tensor:
        out.add_scal_prod(_adjoint_of_word(item.key(), ctx), item.value())

    return out


def _concatenate(a: TensorTensorProduct, otype=rp.FreeTensor):
    """
    Perform an elementwise reduction induced on A \\otimes B by the
    concatenation of words.

    :param a: External tensor product of tensors
    :return: tensor obtained by reducing all pairs of words
    """

    data = defaultdict(lambda: 0)
    for (l, r), v in a.data.items():
        data[l * r] += v

    # noinspection PyArgumentList
    result = otype(data, ctx=a.ctx)
    return result


def _tensor_product_functions(f, g):
    # noinspection PyArgumentList
    def function_product(x: TensorTensorProduct) -> TensorTensorProduct:
        ctx = x.ctx

        result = TensorTensorProduct({}, ctx)
        for (k1, k2), v in x.data.items():
            tk1 = f(rp.FreeTensor((k1, 1), ctx=ctx))
            tk2 = g(rp.FreeTensor((k2, 1), ctx=ctx))
            result.add_scal_prod(TensorTensorProduct((tk1, tk2), ctx), v)

        return result

    return function_product


def _convolve(f, g):
    func = _tensor_product_functions(f, g)

    def convolved(x):
        return _concatenate(func(_adjoint_of_shuffle(x)), otype=type(x))

    return convolved


def _remove_constant(x):
    ctx = x.context
    # noinspection PyArgumentList
    empty_word = rp.TensorKey(width=ctx.width, depth=ctx.depth)
    remover = type(x)((empty_word, x[empty_word]), ctx=ctx)
    return x - remover


Tensor = TypeVar('Tensor')


# noinspection PyPep8Naming
def Log(x: Tensor) -> Tensor:
    """
    Linear function on tensors that agrees with log on the group-like elements.

    This function is the linear extension of the log function defined on the
    group-like elements of the free tensor algebra (or the corresponding subset
    of the shuffle tensor algebra) to the whole algebra. This implementation is
    far from optimal.

    :param x:  Tensor (either a shuffle tensor or free tensor)
    :return: Log(x) with the same type as the input.
    """
    ctx = x.context
    fn = _remove_constant

    out = fn(x)
    sign = False
    for i in range(2, ctx.depth + 1):
        sign = not sign
        fn = _convolve(_remove_constant, fn)

        tmp = fn(x) / i
        if sign:
            out -= tmp
        else:
            out += tmp

    return out
