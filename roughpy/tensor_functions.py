import functools
from collections import defaultdict
from typing import Any, Union, TypeVar

import roughpy as rp


def tensor_word_factory(basis):
    width = basis.width
    depth = basis.depth

    def factory(*args):
        return rp.TensorKey(*args, width=width, depth=depth)

    return factory


class TensorTensorProduct:
    data: dict[tuple[rp.TensorKey], Any]
    ctx: rp.Context

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


def str_ttp(t):
    return " ".join([
        '{',
        *(f"{v}{k}" for k, v in t.items()),
        '}'
    ])

def _concat_product(a, b):
    out = defaultdict(lambda: 0)

    for k1, v1 in a.items():
        for k2, v2 in b.items():
            k = tuple(i * j for i, j in zip(k1, k2))
            v = v1 * v2
            # print("concat prod TT", k, v)
            out[k] += v

    # print("TPEE", str_ttp(a), str_ttp(b), str_ttp(out))
    return {k: v for k, v in out.items() if v != 0}


def _adjoint_of_word(word):
    word_factory = tensor_word_factory(word.basis())
    letters = word.to_letters()
    if not letters:
        return {(word_factory(),) * 2: 1}

    letters_adj = [{(word_factory(letter), word_factory()): 1,
                    (word_factory(), word_factory(letter)): 1}
                   for letter in word.to_letters()]
    return functools.reduce(_concat_product, letters_adj)


def _adjoint_of_shuffle(tensor: Union[rp.FreeTensor, rp.ShuffleTensor]) -> TensorTensorProduct:
    ctx = tensor.context
    out = TensorTensorProduct(defaultdict(lambda: 0), ctx)

    for item in tensor:
        ivalue = item.value()
        prod = TensorTensorProduct(_adjoint_of_word(item.key()), ctx)
        out.add_scal_prod(prod, ivalue)
        # for k, v in prod.items():
        #     out[k] += ivalue * v
        # print("apsi", prod, ivalue)

    # result = TensorTensorProduct(out, ctx=tensor.context)

    # print("aos", out)
    return out


def _concatenate(a, otype=rp.FreeTensor):
    """
    Perform an elementwise reduction induced on A \\otimes B by the concatenation
    of words.

    :param a:
    :return:
    """

    data = defaultdict(lambda: 0)
    for (l, r), v in a.data.items():
        data[l * r] += v

    result = otype(data, ctx=a.ctx)
    # print("conc", a, result)
    return result


def _tensor_product_functions(f, g):
    def function_product(x: TensorTensorProduct) -> TensorTensorProduct:
        ctx = x.ctx

        result = TensorTensorProduct({}, ctx)
        for (k1, k2), v in x.data.items():
            tk1 = f(rp.FreeTensor((k1, 1), ctx=ctx))
            tk2 = g(rp.FreeTensor((k2, 1), ctx=ctx))
            result.add_scal_prod(TensorTensorProduct((tk1, tk2), ctx), v)
            # print("fp", k1, k2, tk1, tk2, result)

        return result

    return function_product


def _convolve(f, g):
    func = _tensor_product_functions(f, g)

    def convolved(x):
        return _concatenate(func(_adjoint_of_shuffle(x)), otype=type(x))

    return convolved


T = TypeVar('T')


def _remove_constant(x):
    ctx = x.context
    empty_word = rp.TensorKey(width=ctx.width, depth=ctx.depth)
    remover = type(x)((empty_word, x[empty_word]), ctx=ctx)
    # print("RC", remover, x)
    return x - remover


# noinspection PyPep8Naming
def Log(x: T) -> T:
    ctx = x.context
    fn = _remove_constant

    out = fn(x)
    # print(out)
    sign = False
    for i in range(2, ctx.depth + 1):
        sign = not sign
        fn = _convolve(_remove_constant, fn)

        tmp = fn(x) / i
        # print(tmp)
        if sign:
            out -= tmp
            # out.sub_scal_div(fn(x), i)
        else:
            out += tmp
            # out.add_scal_div(fn(x), i)

    return out
