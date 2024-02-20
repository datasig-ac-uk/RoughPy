.. _lies:

**************
Lies
**************

^^^^^^^^^^^^^^^^^^^^^
What are Lie elements
^^^^^^^^^^^^^^^^^^^^^

Lie elements live in the free Lie Algebra.
Group-like elements have a one-to-one correspondence with a stream. That is, for every group-like element, there exists a stream where the signature of that stream is the group-like element.
For more information on Lie Algebras, see :cite:t:`reutenauer_free_1993` and :cite:t:`bourbaki_lie_1989`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do Lies fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You will most commonly encounter Lies when taking the log signature of a path. We can use the Dynkin map to transfer between Lies and signatures.

^^^^^^^^^^^^^^^^^^^^^
How to work with Lies
^^^^^^^^^^^^^^^^^^^^^

To create a Lie element, we should start by creating **data**.

::

    >>> import roughpy
    >>> from roughpy import Monomial

    >>> lie_data_x = [
    1 * roughpy.Monomial("x1"),  # channel (1)
    1 * roughpy.Monomial("x2"),  # channel (2)
    1 * roughpy.Monomial("x3"),  # channel (3)
    1 * roughpy.Monomial("x4"),  # channel ([1, 2])
    1 * roughpy.Monomial("x5"),  # channel ([1, 3])
    1 * roughpy.Monomial("x6"),  # channel ([2, 3])
    ]

Here ``roughpy.Monomial`` creates a monomial object (in this case, a single indeterminate),
and multiplying this by 1 makes this monomial into a polynomial.

We can then use our data to create a Lie element. Here we are creating two Lies with Width 3, Depth 2. The first is using the object above. The second is using the same object, but with ``y`` as the indeterminate name.

::

    >>> lie_x = rp.Lie(lie_data_x, width=3, depth=2, dtype=rp.RationalPoly)
    >>> print(f"{lie_x=!s}")
    lie_x={ { 1(x1) }(1) { 1(x2) }(2) { 1(x3) }(3) { 1(x4) }([1,2]) { 1(x5) }([1,3]) { 1(x6) }([2,3]) }

    >>> lie_y = rp.Lie(lie_data_y, width=3, depth=2, dtype=rp.RationalPoly)
    >>> print(f"{lie_y=!s}")
    lie_y={ { 1(y1) }(1) { 1(y2) }(2) { 1(y3) }(3) { 1(y4) }([1,2]) { 1(y5) }([1,3]) { 1(y6) }([2,3]) }


We can also multply Lies together

::

    >>> result = lie_x * lie_y
    >>> print(f"{result=!s}")
    result={ { 1(x1 y2) -1(x2 y1) }([1,2]) { 1(x1 y3) -1(x3 y1) }([1,3]) { 1(x2 y3) -1(x3 y2) }([2,3]) }


If we provide a complementary :doc:`fundamentals.contexts`, then we can use this to create a :doc:`fundamentals.free_tensors`, from our Lie element.

::

    >>> context = rp.get_context(3, 2, rp.RationalPoly)

    >>> tensor_from_lie = context.lie_to_tensor(lie_x).exp()
    >>> print(f"{tensor_from_lie=!s}")
    tensor_from_lie={ { 1() }() { 1(x1) }(1) { 1(x2) }(2) { 1(x3) }(3) { 1/2(x1^2) }(1,1) { 1(x4) 1/2(x1 x2) }(1,2) { 1(x5) 1/2(x1 x3) }(1,3) { -1(x4) 1/2(x1 x2) }(2,1) { 1/2(x2^2) }(2,2) { 1(x6) 1/2(x2 x3) }(2,3) { -1(x5) 1/2(x1 x3) }(3,1) { -1(x6) 1/2(x2 x3) }(3,2) { 1/2(x3^2) }(3,3) }


We can also go the other way, creating a Lie from a tensor. Using the signature of the stream shown ealier in :doc:`fundamentals.streams`, we can create a corresponding Lie element.

::

    >>> lie_from_tensor = context.tensor_to_lie(sig.log())
    >>> print(f"{lie_from_tensor=!s}")
    lie_from_tensor={ 4.5(1) 9(2) }


^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::