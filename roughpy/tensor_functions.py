

import roughpy as rp

import functools

from collections import defaultdict
from typing import Any, Union

def tensor_word_factory(basis):
    width = basis.width
    depth = basis.depth

    def factory(*args):
        return rp.TensorKey(*args, width=width, depth=depth)

    return factory


class TensorTensorProduct:
    data: dict[tuple[rp.TensorKey], Any]
    ctx: rp.Context

    def __init__(self, data, ctx):
        self.data = data
        self.ctx = ctx

    def __str__(self):
        return str(self.data)


def _concat_product(a, b):
    out = defaultdict(lambda: 0)

    for k1, v1 in a.items():
        for k2, v2 in b.items():
            k = tuple(i * j for i, j in zip(k1, k2))
            out[k] += v1*v2
    return out


def _adjoint_of_word(word):
    word_factory = tensor_word_factory(word.basis())
    letters = word.to_letters()
    if not letters:
        return { (word_factory(),)*2: 1 }

    letters_adj = [{(word_factory(letter), word_factory()): 1,
                    (word_factory(), word_factory(letter)): 1}
                   for letter in word.to_letters()]
    return functools.reduce(_concat_product, letters_adj)


def _adjoint_of_shuffle(tensor: Union[rp.FreeTensor, rp.ShuffleTensor]) -> TensorTensorProduct:
    out = defaultdict(lambda: 0)

    for item in tensor:
        for k, v in _adjoint_of_word(item.key()).items():
            out[k] += item.value()*v

    return TensorTensorProduct(out, ctx=tensor.context)


def _concatenate(a):
    """
    Perform an elementwise reduction induced on A \\otimes B by the concatenation
    of words.

    :param a:
    :return:
    """
    result = a.context.zero_lie()

    for (l, r), v in a.items():
        result[l*r] = v

    return result




