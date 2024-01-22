.. _lies:

**************
Lies
**************

.. todo::

    - Elements of the Free Lie Algebra
    - They have a one to one relationship with the group-like elements
    - Hall basis

To create a Lie element, we should start by creating **data**.

::

    lie_data = [
    1 * roughpy.Monomial("x1"),  # channel (1)
    1 * roughpy.Monomial("x2"),  # channel (2)
    1 * roughpy.Monomial("x3"),  # channel (3)
    1 * roughpy.Monomial("x4"),  # channel ([1, 2])
    1 * roughpy.Monomial("x5"),  # channel ([1, 3])
    1 * roughpy.Monomial("x6"),  # channel ([2, 3])
    ]

Here ``roughpy.Monomial`` creates a monomial object (in this case, a single indeterminate),
and multiplying this by 1 makes this monomial into a polynomial.

We can then use our data to create a Lie element. Here we are creating one with Width 3, Depth 2, with polynomial coefficients.

::

    lie = rp.Lie(lie_data, width=3, depth=2, dtype=rp.RationalPoly)


If we provide a complementary :doc:`fundamentals.contexts`, then we can use this to create a :doc:`fundamentals.free_tensors`, from our Lie element.

::

    context = rp.get_context(3, 2, rp.RationalPoly)

    tensor = context.lie_to_tensor(lie)