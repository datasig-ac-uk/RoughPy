.. _free_tensors:

**************
Free Tensors
**************

^^^^^^^^^^^^^^^^^^^^^
What are free tensors
^^^^^^^^^^^^^^^^^^^^^

Free tensors live in a tensor algebra. Equipped with addition, scalar multiplication and a tensor product,
a tensor algebra is a real, non-commutative unital algebra, with unit :math:`\mathbb{1} = (1, 0, 0, ...)`.
It is within the tensor algebra that the signature of a path lives.

For a more accurate definition of the tensor algebra, see :cite:t:`lyons_signature_2024`. See also :cite:t:`reutenauer_free_1993` and :cite:t:`bourbaki_algebra_1998`.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do free tensors fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You will most commonly encounter free tensors by taking the signature of a stream.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How to work with free tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can create a **free tensor** in many ways using RoughPy.

We create a free tensor using data, a width and a depth. The data you use can take many forms, here are some example constructors:

::

    from roughpy import FreeTensor

    tensor1 = FreeTensor([0., 1., 2., 3.], width=3, depth=2)

    tensor2 = FreeTensor([0, 1, 2, 3], width=3, depth=2)

    tensor3 = FreeTensor((1, 1.0), width=2, depth=2)

    k1 = TensorKey(1, width=2, depth=2)
    tensor4 = FreeTensor((k1, 1.0), width=2, depth=2)

    data = [
        {1: 1., 2: 2.},
        {0: 1., 2: 1.}
        ]
    tensor5 = FreeTensor(data, width=2, depth=2)

You can do **arithmetic** with **free tensors**. We can add, subtract, multiply and divide them

::

    t1 = FreeTensor(data1, width=width, depth=depth)
    t2 = FreeTensor(data2, width=width, depth=depth)

    tensors_sum = t1+t2
    # should be the same as FreeTensor(data1 + data2, width=width, depth=depth)

    tensors_difference = t1-t2
    # should be the same as FreeTensor(data1 - data2, width=width, depth=depth)

    tensors_multiplied = t1 * t2

    tensor_scalar_multiplied =  t1 * 2.0
    # should be the same as FreeTensor(2.0 * data1, width=width, depth=depth)

    tensor_scalar_divided = t1 / 2.0
    # should be the same as FreeTensor(data1 / 2.0, width=width, depth=depth)

as well as take their exponential, log, and antipode.

::

    # exponential
    t1.exp()

    #log
    t1.log()

    # antipode
    t1.antipode()

^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::
