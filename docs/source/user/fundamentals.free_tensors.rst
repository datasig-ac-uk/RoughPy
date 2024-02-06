.. _free_tensors:

**************
Free Tensors
**************

^^^^^^^^^^^^^^^^^^^^^
What are free tensors
^^^^^^^^^^^^^^^^^^^^^

Free tensors are

.. todo::
    Finish what are free tensors. Borrow from notebooks (Check: Handwritten digit classigication: the path signature??)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do free tensors fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You will most commonly encounter free tensors by taking the signature of a stream.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How to work with free tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can create a **free tensor** in many ways using `RoughPy`.

We create a free tensor using data, a Width and a Depth. The data you use can take many forms, here are some example constructors:

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

.. todo::

    Include references: (create individual bib file)
        - Reutenauer Free Lie Algebras
        - Bourbaki Algebra I